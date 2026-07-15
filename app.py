
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from ui_components import (
    apply_global_styles,
    render_badges,
    render_footer,
    render_hero,
    render_result_banner,
    render_section_header,
    render_sidebar_brand,
)


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "app_config.json"
BUNDLE_PATH = APP_DIR / "modeles" / "modele_acces_soins_bundle.joblib"
STYLES_PATH = APP_DIR / "assets" / "styles.css"


st.set_page_config(
    page_title="Prédiction du recours formel aux soins",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)




@st.cache_resource
def load_assets() -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bundle = joblib.load(BUNDLE_PATH)
    return config, bundle


def field_label(config: dict[str, Any], feature: str) -> str:
    return config.get("labels", {}).get(feature, feature)


def sync_location(config: dict[str, Any], reset_quarter: bool = False) -> None:
    commune = st.session_state.get("field_Commune")
    locality = config["locality"].get(commune, {})
    quartiers = list(locality.get("quartiers", {}).keys())
    current_quarter = st.session_state.get("field_Quartier")

    if reset_quarter or current_quarter not in quartiers:
        st.session_state.field_Quartier = quartiers[0] if quartiers else None

    quartier = st.session_state.get("field_Quartier")
    quarter_data = locality.get("quartiers", {}).get(quartier, {})
    st.session_state.field_latitude = float(
        quarter_data.get("latitude", locality.get("latitude", -4.325))
    )
    st.session_state.field_longitude = float(
        quarter_data.get("longitude", locality.get("longitude", 15.322))
    )
    st.session_state.zone_display = locality.get(
        "zone", "Zone de résidence non classée"
    )
    st.session_state.prediction_result = None


def commune_changed(config: dict[str, Any]) -> None:
    sync_location(config, reset_quarter=True)


def quarter_changed(config: dict[str, Any]) -> None:
    sync_location(config, reset_quarter=False)


def initialize_state(config: dict[str, Any]) -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    for feature, value in config["defaults"].items():
        key = f"field_{feature}"
        if feature == "_Coordonnées GPS du ménage_latitude":
            continue
        if feature == "_Coordonnées GPS du ménage_longitude":
            continue
        if key not in st.session_state:
            st.session_state[key] = value

    if "field_latitude" not in st.session_state:
        st.session_state.field_latitude = 0.0
    if "field_longitude" not in st.session_state:
        st.session_state.field_longitude = 0.0
    if "zone_display" not in st.session_state:
        st.session_state.zone_display = ""

    sync_location(config, reset_quarter=False)


def load_scenario(config: dict[str, Any], scenario_name: str) -> None:
    scenario = config["scenarios"][scenario_name]["inputs"]
    for feature, value in scenario.items():
        if feature == "_Coordonnées GPS du ménage_latitude":
            st.session_state.field_latitude = float(value)
        elif feature == "_Coordonnées GPS du ménage_longitude":
            st.session_state.field_longitude = float(value)
        else:
            st.session_state[f"field_{feature}"] = value
    sync_location(config, reset_quarter=False)
    st.session_state.prediction_result = None


def select_input(
    config: dict[str, Any],
    feature: str,
    *,
    disabled: bool = False,
    help_text: str | None = None,
) -> str | None:
    options = config["categories"].get(feature, [])
    key = f"field_{feature}"
    current = st.session_state.get(key)

    if current not in options and options:
        st.session_state[key] = options[0]

    return st.selectbox(
        field_label(config, feature),
        options=options,
        key=key,
        disabled=disabled,
        help=help_text,
    )


def render_location(config: dict[str, Any]) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "Commune",
            options=list(config["locality"].keys()),
            key="field_Commune",
            on_change=commune_changed,
            args=(config,),
            help=(
                "La commune détermine la liste des quartiers et les coordonnées "
                "de référence utilisées par le modèle."
            ),
        )

    commune = st.session_state.field_Commune
    quarters = list(
        config["locality"][commune]["quartiers"].keys()
    )
    if st.session_state.get("field_Quartier") not in quarters:
        st.session_state.field_Quartier = quarters[0]

    with col2:
        st.selectbox(
            "Quartier",
            options=quarters,
            key="field_Quartier",
            on_change=quarter_changed,
            args=(config,),
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input(
            "Zone de résidence (information)",
            value=st.session_state.zone_display,
            disabled=True,
        )
    with c2:
        st.number_input(
            "Latitude de référence",
            value=float(st.session_state.field_latitude),
            format="%.7f",
            disabled=True,
        )
    with c3:
        st.number_input(
            "Longitude de référence",
            value=float(st.session_state.field_longitude),
            format="%.7f",
            disabled=True,
        )

    st.map(
        pd.DataFrame(
            {
                "lat": [st.session_state.field_latitude],
                "lon": [st.session_state.field_longitude],
            }
        ),
        zoom=11,
        use_container_width=True,
    )
    st.caption(
        "Les coordonnées correspondent à la médiane des ménages enquêtés "
        "dans le quartier sélectionné. Elles servent de référence prédictive "
        "et ne constituent pas l’adresse exacte du ménage."
    )


def render_feature_columns(
    config: dict[str, Any],
    features: list[str],
    columns: int = 2,
) -> None:
    cols = st.columns(columns)
    for index, feature in enumerate(features):
        with cols[index % columns]:
            select_input(config, feature)


def render_profile(config: dict[str, Any]) -> None:
    profile_features = config["groups"]["profil"]
    # Situation matrimoniale before spouse education for conditional display.
    first_features = [
        feature
        for feature in profile_features
        if feature != "Niveau d’instruction du conjoint/de la conjointe"
    ]
    render_feature_columns(config, first_features, columns=2)

    marital = st.session_state.get("field_Situation matrimoniale")
    if marital in ["Marié(e)", "Union libre"]:
        select_input(
            config,
            "Niveau d’instruction du conjoint/de la conjointe",
        )
    else:
        st.session_state[
            "field_Niveau d’instruction du conjoint/de la conjointe"
        ] = "Non applicable"
        st.info(
            "Le niveau d’instruction du conjoint est automatiquement fixé à "
            "« Non applicable » pour cette situation matrimoniale."
        )


def render_economy(config: dict[str, Any]) -> None:
    insurance_feature = (
        "Au moins un membre du ménage bénéficie-t-il d’une assurance "
        "ou mutuelle de santé ?"
    )
    dependent = set(config["conditional_fields"]["insurance_fields"])
    features = [
        feature
        for feature in config["groups"]["economie"]
        if feature not in dependent
    ]
    render_feature_columns(config, features, columns=2)

    if st.session_state.get(f"field_{insurance_feature}") == "Oui":
        st.markdown("#### Précisions sur la couverture")
        render_feature_columns(
            config,
            config["conditional_fields"]["insurance_fields"],
            columns=2,
        )
    else:
        for feature in dependent:
            st.session_state[f"field_{feature}"] = None
        st.caption(
            "Les questions relatives au type et à l’effet de la couverture "
            "sont masquées lorsque le ménage n’est pas assuré."
        )


def render_environment(config: dict[str, Any]) -> None:
    render_feature_columns(
        config,
        config["groups"]["environnement"],
        columns=2,
    )


def render_supply(config: dict[str, Any]) -> None:
    near_feature = (
        "Existe-t-il une structure de santé formelle proche du domicile ?"
    )
    dependent = set(config["conditional_fields"]["near_structure_fields"])
    general_features = [
        feature
        for feature in config["groups"]["offre"]
        if feature not in dependent and feature != near_feature
    ]

    select_input(config, near_feature)

    if st.session_state.get(f"field_{near_feature}") == "Oui":
        st.markdown("#### Caractéristiques de la structure proche")
        render_feature_columns(
            config,
            config["conditional_fields"]["near_structure_fields"],
            columns=2,
        )
    else:
        for feature in dependent:
            st.session_state[f"field_{feature}"] = None
        st.caption(
            "Les caractéristiques détaillées de la structure sont masquées "
            "lorsque la proximité n’est pas confirmée."
        )

    st.markdown("#### Offre sanitaire du quartier")
    render_feature_columns(config, general_features, columns=2)


def render_need(config: dict[str, Any]) -> None:
    render_feature_columns(
        config,
        config["groups"]["besoin"],
        columns=2,
    )


def render_perception(config: dict[str, Any]) -> None:
    render_feature_columns(
        config,
        config["groups"]["perception"],
        columns=2,
    )


def collect_inputs(config: dict[str, Any]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for feature in config["feature_order"]:
        if feature == "_Coordonnées GPS du ménage_latitude":
            inputs[feature] = float(st.session_state.field_latitude)
        elif feature == "_Coordonnées GPS du ménage_longitude":
            inputs[feature] = float(st.session_state.field_longitude)
        else:
            value = st.session_state.get(f"field_{feature}")
            inputs[feature] = np.nan if value is None else value
    return inputs


def raw_variable_from_encoded(
    encoded_name: str,
    raw_features: list[str],
) -> str:
    core = encoded_name.split("__", 1)[1] if "__" in encoded_name else encoded_name
    if core in raw_features:
        return core

    for raw in sorted(raw_features, key=len, reverse=True):
        if core.startswith(raw + "_"):
            return raw
    return core


@st.cache_resource
def build_explainer(_model: Any) -> Any:
    return shap.TreeExplainer(_model)


def explain_prediction(
    bundle: dict[str, Any],
    row: pd.DataFrame,
) -> pd.DataFrame:
    preprocessor = bundle["preprocessor"]
    transformed = preprocessor.transform(row)
    if hasattr(transformed, "toarray"):
        transformed_dense = transformed.toarray()
    else:
        transformed_dense = np.asarray(transformed)

    explainer = build_explainer(bundle["xgb_model"])
    shap_values = np.asarray(explainer.shap_values(transformed_dense))
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    values = shap_values[0]

    encoded_names = [
        str(name)
        for name in preprocessor.get_feature_names_out()
    ]
    grouped: dict[str, float] = {}
    for encoded_name, contribution in zip(encoded_names, values):
        raw = raw_variable_from_encoded(
            encoded_name,
            bundle["feature_order"],
        )
        grouped[raw] = grouped.get(raw, 0.0) + float(contribution)

    explanation = pd.DataFrame(
        {
            "Variable": list(grouped.keys()),
            "Contribution": list(grouped.values()),
        }
    )
    explanation["Importance"] = explanation["Contribution"].abs()
    explanation["Sens"] = np.where(
        explanation["Contribution"] >= 0,
        "Augmente la probabilité prédite de recours formel",
        "Réduit la probabilité prédite de recours formel",
    )
    return explanation.sort_values(
        "Importance", ascending=False
    ).head(10)


def probability_figure(p_access: float, threshold: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_access * 100,
            number={"suffix": " %", "font": {"size": 44}},
            title={"text": "Probabilité prédite de recours formel"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#087e8b"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#fde8ea"},
                    {"range": [threshold * 100, 100], "color": "#e8f7f1"},
                ],
                "threshold": {
                    "line": {"color": "#d1495b", "width": 4},
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(
        height=330,
        margin=dict(l=35, r=35, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#172033", family="Segoe UI, Arial, sans-serif"),
    )
    return figure


def shap_figure(explanation: pd.DataFrame) -> go.Figure:
    plot_data = explanation.sort_values("Contribution")
    colors = [
        "#1b9e77" if value >= 0 else "#d1495b"
        for value in plot_data["Contribution"]
    ]
    figure = go.Figure(
        go.Bar(
            x=plot_data["Contribution"],
            y=plot_data["Variable"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Contribution SHAP: %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Principales contributions au résultat individuel",
        xaxis_title="Contribution SHAP",
        yaxis_title="",
        height=470,
        margin=dict(l=20, r=20, t=55, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#172033", family="Segoe UI, Arial, sans-serif"),
        xaxis=dict(gridcolor="#E5EBF0", zerolinecolor="#B7C4CF"),
        yaxis=dict(tickfont=dict(color="#172033")),
    )
    return figure


def vulnerability_level(p_access: float) -> str:
    if p_access < 0.30:
        return "Vulnérabilité élevée"
    if p_access < 0.50:
        return "Vulnérabilité modérée"
    if p_access < 0.70:
        return "Profil favorable modéré"
    return "Profil favorable élevé"


def run_prediction(
    config: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    inputs = collect_inputs(config)
    row = pd.DataFrame(
        [[inputs[feature] for feature in bundle["feature_order"]]],
        columns=bundle["feature_order"],
    )

    transformed = bundle["preprocessor"].transform(row)
    p_xgb = float(
        bundle["xgb_model"].predict_proba(transformed)[0, 1]
    )
    p_rf = float(
        bundle["rf_model"].predict_proba(transformed)[0, 1]
    )
    threshold = float(config["threshold"])
    predicted_class = int(p_xgb >= threshold)
    profile = (
        "Ménage favorable"
        if predicted_class == 1
        else "Ménage vulnérable"
    )

    try:
        explanation = explain_prediction(bundle, row)
        explanation_error = None
    except Exception as exc:
        explanation = None
        explanation_error = str(exc)

    return {
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "inputs": inputs,
        "p_access": p_xgb,
        "p_nonaccess": 1.0 - p_xgb,
        "p_rf": p_rf,
        "predicted_class": predicted_class,
        "profile": profile,
        "level": vulnerability_level(p_xgb),
        "agreement": (p_xgb >= threshold) == (p_rf >= threshold),
        "explanation": explanation,
        "explanation_error": explanation_error,
    }


def render_result(
    config: dict[str, Any],
    bundle: dict[str, Any],
    result: dict[str, Any],
) -> None:
    favorable = result["predicted_class"] == 1
    render_result_banner(
        profile=result["profile"],
        level=result["level"],
        probability=result["p_access"],
        favorable=favorable,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recours formel", f"{result['p_access']:.1%}")
    m2.metric("Non-recours", f"{result['p_nonaccess']:.1%}")
    m3.metric("Seuil officiel", f"{config['threshold']:.0%}")
    m4.metric(
        "Accord XGBoost / RF",
        "Oui" if result["agreement"] else "Non",
    )

    left, right = st.columns([1.0, 1.15])
    with left:
        st.plotly_chart(
            probability_figure(
                result["p_access"],
                config["threshold"],
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Le classement officiel repose sur le XGBoost. Le Random Forest "
            "est utilisé comme contrôle complémentaire de robustesse."
        )
        st.info(
            f"Probabilité Random Forest : {result['p_rf']:.1%}"
        )
        if not result["agreement"]:
            st.warning(
                "Les deux modèles aboutissent à des classes différentes. "
                "Le résultat doit être interprété avec une prudence renforcée."
            )

    with right:
        if result["explanation"] is not None:
            explanation = result["explanation"].copy()
            explanation["Variable"] = explanation["Variable"].map(
                lambda value: field_label(config, value)
            )
            st.plotly_chart(
                shap_figure(explanation),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.warning(
                "L’explication SHAP n’a pas pu être calculée : "
                + str(result["explanation_error"])
            )

    with st.expander("Interprétation détaillée des facteurs"):
        if result["explanation"] is not None:
            table = result["explanation"].copy()
            table["Variable"] = table["Variable"].map(
                lambda value: field_label(config, value)
            )
            table["Contribution"] = table["Contribution"].round(4)
            st.dataframe(
                table[["Variable", "Contribution", "Sens"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Une contribution SHAP décrit la manière dont la variable "
                "oriente la prédiction du modèle. Elle ne prouve pas une "
                "relation causale."
            )

    report = {
        "date_prediction": result["timestamp"],
        "modele_principal": bundle["model_name"],
        "modele_secondaire": bundle["secondary_model_name"],
        "seuil": config["threshold"],
        "probabilite_recours_formel": result["p_access"],
        "probabilite_non_recours": result["p_nonaccess"],
        "probabilite_random_forest": result["p_rf"],
        "classe_predite": (
            "Recours formel probable"
            if result["predicted_class"] == 1
            else "Non-recours probable"
        ),
        "profil_menage": result["profile"],
        "niveau": result["level"],
        "accord_modeles": result["agreement"],
        "entrees": result["inputs"],
    }
    st.download_button(
        "Télécharger le rapport de prédiction (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="rapport_prediction_recours_soins.json",
        mime="application/json",
        use_container_width=True,
    )


def main() -> None:
    apply_global_styles(STYLES_PATH)
    config, bundle = load_assets()
    initialize_state(config)

    with st.sidebar:
        render_sidebar_brand()
        st.markdown("#### Performance du modèle")

        metrics = config["model_metrics"]["xgboost"]
        st.metric("AUC-ROC du XGBoost", f"{metrics['auc_test']:.3f}")
        st.metric("Exactitude sur test", f"{metrics['accuracy_test']:.1%}")
        st.metric("Score de Brier", f"{metrics['brier']:.3f}")

        st.markdown("---")
        st.markdown("### Scénarios de démonstration")
        for scenario_name in config["scenarios"]:
            if st.button(
                scenario_name,
                key=f"scenario_{scenario_name}",
                use_container_width=True,
            ):
                load_scenario(config, scenario_name)
                st.rerun()

        if st.button("Réinitialiser le formulaire", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("field_") or key in [
                    "zone_display",
                    "prediction_result",
                ]:
                    del st.session_state[key]
            initialize_state(config)
            st.rerun()

        st.markdown("---")
        with st.expander("Méthode et limites"):
            st.markdown(
                """
                - Base analytique : **1 000 ménages**.
                - Cible : recours effectif à une structure formelle.
                - Modèle principal : **XGBoost optimisé**.
                - Seuil de classement : **0,50**.
                - Le résultat est probabiliste et non causal.
                - L’application n’est ni un diagnostic médical ni une décision
                  administrative automatique.
                - Une utilisation hors de Kinshasa nécessite une validation
                  externe et un réentraînement.
                """
            )

        if st.session_state.prediction_history:
            with st.expander("Historique de la session"):
                history = pd.DataFrame(
                    st.session_state.prediction_history
                )
                st.dataframe(
                    history,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Exporter l’historique (CSV)",
                    data=history.to_csv(index=False).encode("utf-8-sig"),
                    file_name="historique_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    render_hero(
        config["application_title"],
        config["application_subtitle"],
    )
    render_badges(
        [
            "XGBoost principal",
            "Random Forest de contrôle",
            "54 prédicteurs",
            "1 000 ménages analysés",
        ]
    )

    render_section_header(
        "1",
        "Renseignement du ménage",
        "Complétez les informations disponibles. Les champs sont organisés par thème pour faciliter la saisie.",
    )
    tabs = st.tabs(
        [
            "📍 Localisation",
            "👥 Profil",
            "💰 Situation économique",
            "🌍 Environnement",
            "🏥 Offre et mobilité",
            "🩺 Besoin de soins",
            "⭐ Perception des services",
        ]
    )

    with tabs[0]:
        with st.container(border=True):
            render_location(config)
    with tabs[1]:
        with st.container(border=True):
            render_profile(config)
    with tabs[2]:
        with st.container(border=True):
            render_economy(config)
    with tabs[3]:
        with st.container(border=True):
            render_environment(config)
    with tabs[4]:
        with st.container(border=True):
            render_supply(config)
    with tabs[5]:
        with st.container(border=True):
            render_need(config)
    with tabs[6]:
        with st.container(border=True):
            render_perception(config)

    render_section_header(
        "2",
        "Calcul de la prédiction",
        "Le calcul applique automatiquement le prétraitement et les deux modèles enregistrés.",
    )
    with st.container(border=True):
        st.caption(
            "Le formulaire utilise uniquement des variables disponibles avant ou "
            "au moment de la décision de recours. Les variables postérieures au "
            "recours ont été exclues pour éviter la fuite de cible."
        )

        if st.button(
            "Calculer la probabilité de recours formel",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Prétraitement des données et calcul de la prédiction…"):
                result = run_prediction(config, bundle)
                st.session_state.prediction_result = result
                st.session_state.prediction_history.append(
                    {
                        "Date": result["timestamp"],
                        "Commune": result["inputs"]["Commune"],
                        "Quartier": result["inputs"]["Quartier"],
                        "Probabilité de recours": round(
                            result["p_access"], 4
                        ),
                        "Profil": result["profile"],
                        "Accord des modèles": (
                            "Oui" if result["agreement"] else "Non"
                        ),
                    }
                )
                st.session_state.prediction_history = (
                    st.session_state.prediction_history[-20:]
                )

    if st.session_state.prediction_result:
        render_section_header(
            "3",
            "Résultat de la prédiction",
            "La probabilité, la classe opérationnelle et les principaux facteurs sont présentés ci-dessous.",
        )
        render_result(
            config,
            bundle,
            st.session_state.prediction_result,
        )

    render_footer()


if __name__ == "__main__":
    main()
