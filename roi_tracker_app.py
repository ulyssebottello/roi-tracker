"""
ROI Tracker Chatbot - Outil d'analyse ROI pour chatbots
Technologie: Python + Streamlit
Objectif: Analyser les comportements trackés sur un site web pour calculer le ROI d'une solution chatbot
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter

# Configuration de la page
st.set_page_config(
    page_title="ROI Tracker - Analyse Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour l'interface
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Fonction utilitaire pour le formatage des euros avec séparateurs de milliers
def format_euro(amount: float) -> str:
    """Formate un montant en euros avec séparateurs de milliers (espaces)"""
    return f"{amount:,.2f}€".replace(',', ' ')

# Classes pour la gestion des données
class ROIDataProcessor:
    """Classe pour traiter et analyser les données ROI"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.processed_data = None
        self.consolidated_data = None
        self.events_analysis = {}
        self.filtered_data = None
        self.date_filter_applied = False
        
    def validate_data_structure(self) -> Tuple[bool, str]:
        """Valide la structure du DataFrame"""
        # Support de deux formats de données
        # Format 1: name, values, date (format clé:valeur)
        # Format 2: name, date + colonnes value_* (format colonnes séparées)
        
        required_base = ['name', 'date']
        
        if not all(col in self.df.columns for col in required_base):
            missing_cols = [col for col in required_base if col not in self.df.columns]
            return False, f"Colonnes de base manquantes: {missing_cols}"
        
        # Vérification du format
        has_values_column = 'values' in self.df.columns
        has_value_columns = any(col.startswith('value_') for col in self.df.columns)
        
        if not has_values_column and not has_value_columns:
            return False, "Aucune colonne de valeurs trouvée (attendu: 'values' ou 'value_*')"
        
        if self.df.empty:
            return False, "Le fichier est vide"
            
        return True, "Structure valide"
    
    def parse_values_column(self) -> pd.DataFrame:
        """Parse les colonnes de valeurs selon le format détecté"""
        consolidated_data = {}
        
        # Détection du format de données
        has_values_column = 'values' in self.df.columns
        has_value_columns = any(col.startswith('value_') for col in self.df.columns)
        
        for idx, row in self.df.iterrows():
            # Clé unique pour identifier un événement (nom + date + conversation)
            event_key = (
                row['name'], 
                row['date'], 
                row.get('conversation_id', row.get('conversationId', ''))
            )
            
            # Initialiser l'événement s'il n'existe pas
            if event_key not in consolidated_data:
                consolidated_data[event_key] = {
                    'name': row['name'],
                    'date': row['date'],
                    'conversation_id': row.get('conversation_id', row.get('conversationId', '')),
                    'values': {}
                }
            
            if has_values_column:
                # Format 1: colonne 'values' avec format clé:valeur
                values_str = str(row['values'])
                
                # Parse le format clé:valeur
                if ':' in values_str:
                    parts = values_str.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        
                        # Tentative de conversion en nombre pour les prix
                        numeric_value = None
                        try:
                            numeric_value = float(value)
                        except ValueError:
                            pass
                        
                        consolidated_data[event_key]['values'][key] = {
                            'text': value,
                            'numeric': numeric_value
                        }
                        
            elif has_value_columns:
                # Format 2: colonnes value_* séparées
                value_columns = [col for col in self.df.columns if col.startswith('value_')]
                
                has_values = False
                
                for value_col in value_columns:
                    value_content = row[value_col]
                    
                    # Si la valeur est vide/NaN, continuer
                    if pd.isna(value_content) or str(value_content).strip() == '':
                        continue
                    
                    has_values = True
                    
                    # Extraction de la clé depuis le nom de colonne (value_custom -> custom)
                    key_from_column = value_col.replace('value_', '')
                    
                    # Parse du contenu - gestion des multiples paires séparées par |
                    content_str = str(value_content).strip()
                    
                    # Vérifier s'il y a des multiples paires séparées par |
                    if '|' in content_str and ':' in content_str:
                        # Format multiple : clé1:valeur1|clé2:valeur2|...
                        pairs = content_str.split('|')
                        for pair in pairs:
                            pair = pair.strip()
                            if ':' in pair:
                                parts = pair.split(':', 1)
                                if len(parts) == 2:
                                    key = parts[0].strip()
                                    value = parts[1].strip()
                                    
                                    # Tentative de conversion en nombre pour les prix
                                    numeric_value = None
                                    try:
                                        # Nettoyer la valeur pour extraction numérique (enlever €, espaces, etc.)
                                        clean_value = value.replace('€', '').replace(',', '.').strip()
                                        numeric_value = float(clean_value)
                                    except ValueError:
                                        pass
                                    
                                    consolidated_data[event_key]['values'][key] = {
                                        'text': value,
                                        'numeric': numeric_value
                                    }
                    elif ':' in content_str:
                        # Format simple : clé:valeur
                        parts = content_str.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                        else:
                            key = key_from_column
                            value = content_str
                    else:
                        # Pas de format clé:valeur, utiliser le nom de colonne comme clé
                        key = key_from_column
                        value = content_str
                    
                    # Si on est dans les cas simples, traiter la valeur
                    if not ('|' in content_str and ':' in content_str):
                        # Tentative de conversion en nombre pour les prix
                        numeric_value = None
                        try:
                            # Nettoyer la valeur pour extraction numérique (enlever €, espaces, etc.)
                            clean_value = value.replace('€', '').replace(',', '.').strip()
                            numeric_value = float(clean_value)
                        except ValueError:
                            pass
                        
                        consolidated_data[event_key]['values'][key] = {
                            'text': value,
                            'numeric': numeric_value
                        }
                
                # Si aucune valeur n'a été trouvée, marquer comme no_value
                if not has_values:
                    consolidated_data[event_key]['values']['no_value'] = {
                        'text': '',
                        'numeric': None
                    }
        
        # Conversion en format DataFrame consolidé
        final_data = []
        for event_key, event_data in consolidated_data.items():
            # Créer une ligne consolidée avec toutes les valeurs
            row_data = {
                'name': event_data['name'],
                'date': event_data['date'],
                'conversation_id': event_data['conversation_id']
            }
            
            # Ajouter toutes les valeurs comme colonnes séparées
            for value_key, value_info in event_data['values'].items():
                row_data[f'{value_key}_text'] = value_info['text']
                row_data[f'{value_key}_numeric'] = value_info['numeric']
            
            final_data.append(row_data)
        
        consolidated_df = pd.DataFrame(final_data)
        
        # Créer également un format "long" pour compatibilité avec l'analyse existante
        long_format_data = []
        for event_key, event_data in consolidated_data.items():
            for value_key, value_info in event_data['values'].items():
                long_format_data.append({
                    'name': event_data['name'],
                    'date': event_data['date'],
                    'value_key': value_key,
                    'value_text': value_info['text'],
                    'value_numeric': value_info['numeric'],
                    'conversation_id': event_data['conversation_id']
                })
        
        # Stocker les deux formats
        self.consolidated_data = consolidated_df
        return pd.DataFrame(long_format_data)
    
    def detect_value_types(self) -> Dict[str, str]:
        """Détecte automatiquement les types de valeurs"""
        if self.processed_data is None:
            self.processed_data = self.parse_values_column()
            
        type_mapping = {}
        
        for key in self.processed_data['value_key'].unique():
            key_lower = key.lower()
            if 'price' in key_lower or 'prix' in key_lower:
                type_mapping[key] = 'price'
            elif 'product' in key_lower or 'produit' in key_lower:
                type_mapping[key] = 'product'
            else:
                type_mapping[key] = 'custom'
                
        return type_mapping
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Retourne la plage de dates disponible dans les données"""
        if self.processed_data is None:
            self.processed_data = self.parse_values_column()
            
        # Conversion de la colonne date
        self.processed_data['date'] = pd.to_datetime(self.processed_data['date'], errors='coerce')
        
        # Normaliser les timezones pour éviter les erreurs de comparaison
        if self.processed_data['date'].dt.tz is not None:
            self.processed_data['date'] = self.processed_data['date'].dt.tz_convert(None)
        
        min_date = self.processed_data['date'].min()
        max_date = self.processed_data['date'].max()
        
        return min_date, max_date
    
    def apply_date_filter(self, start_date: datetime, end_date: datetime):
        """Applique un filtre de date aux données"""
        if self.processed_data is None:
            self.processed_data = self.parse_values_column()
            
        # Conversion de la colonne date si pas déjà fait
        self.processed_data['date'] = pd.to_datetime(self.processed_data['date'], errors='coerce')
        
        # Normaliser les timezones pour éviter les erreurs de comparaison
        # Convertir en timezone naive si les dates ont une timezone
        if self.processed_data['date'].dt.tz is not None:
            self.processed_data['date'] = self.processed_data['date'].dt.tz_convert(None)
        
        # S'assurer que les dates de filtrage sont aussi timezone naive
        if start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        if end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)
        
        # Filtrer les données selon la période
        mask = (
            (self.processed_data['date'] >= start_date) & 
            (self.processed_data['date'] <= end_date)
        )
        
        self.filtered_data = self.processed_data[mask].copy()
        self.date_filter_applied = True
        
        # Filtrer aussi les données consolidées si elles existent
        if self.consolidated_data is not None:
            consolidated_dates = pd.to_datetime(self.consolidated_data['date'], errors='coerce')
            
            # Normaliser les timezones pour les données consolidées aussi
            if consolidated_dates.dt.tz is not None:
                consolidated_dates = consolidated_dates.dt.tz_convert(None)
                
            consolidated_mask = (
                (consolidated_dates >= start_date) & 
                (consolidated_dates <= end_date)
            )
            self.consolidated_data = self.consolidated_data[consolidated_mask].copy()
    
    def get_active_data(self) -> pd.DataFrame:
        """Retourne les données actives (filtrées ou complètes)"""
        if self.date_filter_applied and self.filtered_data is not None:
            return self.filtered_data
        else:
            if self.processed_data is None:
                self.processed_data = self.parse_values_column()
            return self.processed_data
    
    def get_global_metrics(self) -> Dict:
        """Calcule les métriques globales"""
        # Utiliser les données actives (filtrées ou complètes)
        active_data = self.get_active_data()
        
        # Conversion de la colonne date
        active_data['date'] = pd.to_datetime(active_data['date'], errors='coerce')
        
        # Normaliser les timezones pour éviter les erreurs de comparaison
        if active_data['date'].dt.tz is not None:
            active_data['date'] = active_data['date'].dt.tz_convert(None)
        
        total_events = len(active_data)
        unique_events = active_data['name'].nunique()
        date_range = (
            active_data['date'].min(),
            active_data['date'].max()
        )
        
        # Répartition par type d'événement
        event_distribution = active_data['name'].value_counts()
        
        # Répartition par type de valeur
        value_type_distribution = active_data['value_key'].value_counts()
        
        return {
            'total_events': total_events,
            'unique_events': unique_events,
            'date_range': date_range,
            'event_distribution': event_distribution,
            'value_type_distribution': value_type_distribution
        }
    
    def analyze_selected_events(self, selected_events: List[str]) -> Dict:
        """Analyse les événements sélectionnés"""
        # Utiliser les données actives (filtrées ou complètes)
        active_data = self.get_active_data()
            
        # Filtrage des données
        filtered_data = active_data[
            active_data['value_key'].isin(selected_events)
        ].copy()
        
        if filtered_data.empty:
            return {'error': 'Aucune donnée trouvée pour les événements sélectionnés'}
        
        analysis = {}
        type_mapping = self.detect_value_types()
        
        # Analyse par type
        for event in selected_events:
            event_data = filtered_data[filtered_data['value_key'] == event]
            event_type = type_mapping.get(event, 'custom')
            
            if event_type == 'price':
                analysis[event] = self._analyze_price_data(event_data, event)
            elif event_type == 'product':
                analysis[event] = self._analyze_product_data(event_data, event)
            else:
                analysis[event] = self._analyze_custom_data(event_data, event)
                
        return analysis
    
    def _analyze_price_data(self, data: pd.DataFrame, event: str) -> Dict:
        """Analyse spécialisée pour les données de prix"""
        # Filtrer les valeurs numériques valides
        price_data = data.dropna(subset=['value_numeric'])
        
        if price_data.empty:
            return {'type': 'price', 'error': 'Aucune donnée de prix valide'}
        
        total_revenue = price_data['value_numeric'].sum()
        avg_transaction = price_data['value_numeric'].mean()
        transaction_count = len(price_data)
        
        # Évolution temporelle du CA - grouper par jour pour éviter les points multiples
        # Grouper par jour et sommer les revenus
        daily_revenue = price_data.groupby(
            price_data['date'].dt.date
        )['value_numeric'].sum().reset_index()
        daily_revenue.columns = ['date', 'daily_revenue']
        
        # Convertir la date en datetime et trier
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        daily_revenue = daily_revenue.sort_values('date')
        
        # Calculer le CA cumulé par jour
        daily_revenue['cumulative_revenue'] = daily_revenue['daily_revenue'].cumsum()
        
        return {
            'type': 'price',
            'total_revenue': total_revenue,
            'avg_transaction': avg_transaction,
            'transaction_count': transaction_count,
            'temporal_data': daily_revenue,
            'distribution': price_data['value_numeric'].value_counts().sort_index()
        }
    
    def _analyze_product_data(self, data: pd.DataFrame, event: str) -> Dict:
        """Analyse spécialisée pour les données produit"""
        product_distribution = data['value_text'].value_counts()
        
        # Évolution temporelle par produit
        temporal_data = data.groupby(['date', 'value_text']).size().unstack(fill_value=0)
        
        return {
            'type': 'product',
            'distribution': product_distribution,
            'temporal_data': temporal_data,
            'top_products': product_distribution.head(10)
        }
    
    def _analyze_custom_data(self, data: pd.DataFrame, event: str) -> Dict:
        """Analyse générale pour les données personnalisées"""
        distribution = data['value_text'].value_counts()
        
        return {
            'type': 'custom',
            'distribution': distribution,
            'count': len(data)
        }


def configure_date_axis(fig, dates_data):
    """Configure intelligemment l'axe des dates pour éviter la répétition des labels"""
    num_days = len(dates_data) if hasattr(dates_data, '__len__') else len(set(dates_data))
    
    if num_days <= 2:
        # Pour 1-2 jours, afficher les dates disponibles sans répétition
        unique_dates = sorted(set(dates_data))
        fig.update_xaxes(
            tickmode='array',
            tickvals=unique_dates,
            ticktext=[pd.to_datetime(d).strftime('%d/%m/%Y') for d in unique_dates],
            tickformat='%d/%m/%Y'
        )
    elif num_days <= 7:
        # Pour une semaine, un tick par jour
        fig.update_xaxes(
            tickformat='%d/%m',
            tickmode='linear',
            dtick='D1'
        )
    else:
        # Pour plus de 7 jours, espacement automatique intelligent
        fig.update_xaxes(
            tickformat='%d/%m',
            tickmode='auto',
            nticks=min(10, num_days)
        )
    
    return fig


def load_and_validate_data(uploaded_file) -> Tuple[Optional[ROIDataProcessor], str]:
    """Charge et valide le fichier uploadé"""
    try:
        # Lecture du fichier CSV
        df = pd.read_csv(uploaded_file)
        
        # Création du processeur de données
        processor = ROIDataProcessor(df)
        
        # Validation de la structure
        is_valid, message = processor.validate_data_structure()
        
        if not is_valid:
            return None, f"❌ Erreur de validation: {message}"
        
        return processor, f"✅ Fichier chargé avec succès: {len(df)} lignes"
        
    except Exception as e:
        return None, f"❌ Erreur lors du chargement: {str(e)}"


def display_period_picker(processor: ROIDataProcessor) -> Tuple[datetime, datetime]:
    """Affiche le sélecteur de période et retourne les dates sélectionnées"""
    # Obtenir la plage de dates disponible dans le fichier
    min_date, max_date = processor.get_date_range()
    
    if pd.isna(min_date) or pd.isna(max_date):
        st.error("❌ Impossible de déterminer la plage de dates du fichier")
        return None, None
    
    # Convertir en dates Python pour Streamlit
    min_date_py = min_date.date()
    max_date_py = max_date.date()
    
    # Interface de sélection simple
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Date de début",
            value=min_date_py,
            min_value=min_date_py,
            max_value=max_date_py,
            key="start_date"
        )
    
    with col2:
        end_date = st.date_input(
            "Date de fin", 
            value=max_date_py,
            min_value=min_date_py,
            max_value=max_date_py,
            key="end_date"
        )
    
    # Validation de la sélection
    if start_date > end_date:
        st.error("❌ La date de début doit être antérieure à la date de fin")
        return None, None
    
    # Conversion en datetime pour le filtrage
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    return start_datetime, end_datetime


def display_global_metrics(processor: ROIDataProcessor, total_genii_conversations=None, conversion_events=None):
    """Affiche les métriques globales (Niveau 1)"""
    st.markdown('<div class="section-header">📊 Vue Généraliste - Métriques Globales</div>', 
                unsafe_allow_html=True)
    
    metrics = processor.get_global_metrics()
    
    # Calcul du taux de conversion Genii si les paramètres sont fournis
    genii_conversion_rate = None
    conversion_count = 0
    total_conversion_events = 0
    
    if total_genii_conversations and conversion_events:
        active_data = processor.get_active_data()
        
        # Gestion de liste ou d'un seul événement
        events_list = conversion_events if isinstance(conversion_events, list) else [conversion_events]
        
        # Compter le nombre total d'événements de conversion (pour information)
        total_conversion_events = len(active_data[active_data['name'].isin(events_list)])
        
        # Compter les conversations uniques avec conversion (logique cohérente avec tracking)
        conversations_with_conversion = active_data[
            active_data['name'].isin(events_list)
        ]['conversation_id'].unique()
        
        # Filtrer les conversations avec ID valide
        valid_conversations = [
            conv_id for conv_id in conversations_with_conversion 
            if not (pd.isna(conv_id) or conv_id == '')
        ]
        
        conversion_count = len(valid_conversations)
        
        if total_genii_conversations > 0:
            genii_conversion_rate = (conversion_count / total_genii_conversations) * 100
    
    # Métriques principales - ajuster le nombre de colonnes selon la présence du taux de conversion
    if genii_conversion_rate is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
    else:
        col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total d'événements",
            value=f"{metrics['total_events']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Types d'événements uniques",
            value=metrics['unique_events'],
            delta=None
        )
    
    with col3:
        date_range = metrics['date_range']
        if pd.notna(date_range[0]) and pd.notna(date_range[1]):
            duration = (date_range[1] - date_range[0]).days
            st.metric(
                label="Période d'analyse",
                value=f"{duration} jours",
                delta=None
            )
    
    with col4:
        avg_events_per_day = metrics['total_events'] / max(duration, 1) if 'duration' in locals() else 0
        st.metric(
            label="Événements/jour (moy.)",
            value=f"{avg_events_per_day:.1f}",
            delta=None
        )
    
    # Afficher le taux de conversion Genii si calculé
    if genii_conversion_rate is not None:
        with col5:
            st.metric(
                label="Taux de Conversion Genii",
                value=f"{genii_conversion_rate:.2f}%",
                delta=f"{conversion_count} conversations"
            )
            
            # Information supplémentaire sur les événements vs conversations
            if total_conversion_events > conversion_count:
                st.caption(f"ℹ️ {total_conversion_events} événements au total • {conversion_count} conversations uniques")
            else:
                st.caption(f"ℹ️ Basé sur {conversion_count} conversations uniques")
    
    # Graphiques de répartition
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Répartition par type d'événement")
        if not metrics['event_distribution'].empty:
            fig_events = px.pie(
                values=metrics['event_distribution'].values,
                names=metrics['event_distribution'].index,
                title="Distribution des événements"
            )
            fig_events.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_events, use_container_width=True)
    
    with col2:
        st.subheader("Répartition par type de valeur")
        if not metrics['value_type_distribution'].empty:
            fig_values = px.pie(
                values=metrics['value_type_distribution'].values,
                names=metrics['value_type_distribution'].index,
                title="Distribution des types de valeurs"
            )
            fig_values.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_values, use_container_width=True)
    
    # Évolution temporelle
    st.subheader("Évolution temporelle des événements")
    if processor.processed_data is not None:
        # Grouper par jour complet pour éviter les points multiples dans la journée
        daily_events = processor.processed_data.groupby(
            processor.processed_data['date'].dt.date
        ).size().reset_index()
        daily_events.columns = ['date', 'count']
        
        # Convertir la date en datetime pour un meilleur affichage
        daily_events['date'] = pd.to_datetime(daily_events['date'])
        
        fig_timeline = px.line(
            daily_events, 
            x='date', 
            y='count',
            title="Nombre d'événements par jour",
            labels={'count': 'Nombre d\'événements', 'date': 'Date'}
        )
        # Configurer l'axe des dates intelligemment
        configure_date_axis(fig_timeline, daily_events['date'])
        st.plotly_chart(fig_timeline, use_container_width=True)


def display_event_explorer(processor: ROIDataProcessor):
    """Affiche l'explorateur d'événements (Niveau 2)"""
    st.markdown('<div class="section-header">🔍 Explorateur d\'Événements</div>', 
                unsafe_allow_html=True)
    
    # Utiliser les données actives (filtrées ou complètes)
    active_data = processor.get_active_data()
    
    # Liste des noms d'événements disponibles (pas les value_key, mais les 'name')
    available_event_names = sorted(active_data['name'].unique())
    
    st.subheader("Sélection des événements à analyser")
    
    # Interface de sélection par nom d'événement
    selected_event_names = []
    
    # Affichage de tous les événements avec checkbox
    st.write("**📋 Événements disponibles**")
    
    # Organisation en colonnes pour une meilleure présentation
    cols = st.columns(2)
    col_idx = 0
    
    for event_name in available_event_names:
        with cols[col_idx % 2]:
            # Comptage CORRECT du nombre d'événements uniques (pas de doublons dus au format long)
            # Compter les combinaisons uniques (name, date, conversation_id)
            event_data = active_data[active_data['name'] == event_name]
            event_count = len(event_data.drop_duplicates(subset=['name', 'date', 'conversation_id']))
            
            if st.checkbox(f"{event_name} ({event_count} événements)", key=f"event_{event_name}"):
                selected_event_names.append(event_name)
        
        col_idx += 1
    
    # Analyse automatique des événements sélectionnés (pas besoin de bouton)
    if selected_event_names:
        st.markdown("---")
        display_event_name_analysis(processor, selected_event_names)


def display_event_name_analysis(processor: ROIDataProcessor, selected_event_names: List[str]):
    """Affiche l'analyse des événements sélectionnés par nom"""
    st.markdown('<div class="section-header">📈 Analyse des Événements Sélectionnés</div>', 
                unsafe_allow_html=True)
    
    # Utiliser les données actives (filtrées ou complètes)
    active_data = processor.get_active_data()
    
    # Filtrage des données par nom d'événement
    filtered_data = active_data[
        active_data['name'].isin(selected_event_names)
    ].copy()
    
    if filtered_data.empty:
        st.warning("Aucune donnée trouvée pour les événements sélectionnés")
        return
    
    # Conversion des dates si pas déjà fait
    filtered_data['date'] = pd.to_datetime(filtered_data['date'], errors='coerce')
    
    # Normaliser les timezones pour éviter les erreurs de comparaison
    if filtered_data['date'].dt.tz is not None:
        filtered_data['date'] = filtered_data['date'].dt.tz_convert(None)
    
    # === MÉTRIQUES PRINCIPALES ===
    # Compter les événements uniques (éviter les doublons dus au format long)
    unique_events = filtered_data.drop_duplicates(subset=['name', 'date', 'conversation_id'])
    total_events = len(unique_events)
    unique_days = filtered_data['date'].dt.date.nunique()
    avg_events_per_day = total_events / max(unique_days, 1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Nombre total d'événements",
            value=f"{total_events:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Jours d'activité",
            value=unique_days,
            delta=None
        )
    
    with col3:
        st.metric(
            label="Événements/jour (moy.)",
            value=f"{avg_events_per_day:.1f}",
            delta=None
        )
    
    # === ÉVOLUTION TEMPORELLE DES ÉVÉNEMENTS ===
    st.subheader("📊 Évolution du nombre d'événements")
    
    # Comptage par jour et par nom d'événement (utiliser unique_events pour éviter doublons)
    daily_events = unique_events.groupby([
        unique_events['date'].dt.date, 'name'
    ]).size().unstack(fill_value=0).reset_index()
    
    # Si un seul type d'événement, graphique simple
    if len(selected_event_names) == 1:
        daily_simple = unique_events.groupby(
            unique_events['date'].dt.date
        ).size().reset_index()
        daily_simple.columns = ['date', 'count']
        
        # Convertir la date en datetime pour un meilleur affichage
        daily_simple['date'] = pd.to_datetime(daily_simple['date'])
        
        fig_timeline = px.line(
            daily_simple,
            x='date',
            y='count',
            title=f"Évolution - {selected_event_names[0]}",
            labels={'count': 'Nombre d\'événements', 'date': 'Date'}
        )
        # Configurer l'axe des dates intelligemment
        configure_date_axis(fig_timeline, daily_simple['date'])
        
    else:
        # Plusieurs événements : graphique empilé
        # Convertir la date en datetime pour un meilleur affichage
        daily_events['date'] = pd.to_datetime(daily_events['date'])
        
        daily_melted = daily_events.melt(
            id_vars=['date'], 
            var_name='event_name', 
            value_name='count'
        )
        
        fig_timeline = px.area(
            daily_melted,
            x='date',
            y='count',
            color='event_name',
            title="Évolution par type d'événement",
            labels={'count': 'Nombre d\'événements', 'date': 'Date'}
        )
        # Configurer l'axe des dates intelligemment
        configure_date_axis(fig_timeline, daily_events['date'])
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # === DÉTECTION ET ANALYSE DES PRIX ===
    # Chercher les événements qui ont des valeurs de prix
    price_data = filtered_data[
        (filtered_data['value_key'].str.contains('price', case=False, na=False)) |
        (filtered_data['value_key'].str.contains('prix', case=False, na=False))
    ].copy()
    
    price_data = price_data.dropna(subset=['value_numeric'])
    
    if not price_data.empty:
        st.markdown("---")
        st.subheader("💰 Analyse du Chiffre d'Affaires")
        
        # Métriques CA
        total_revenue = price_data['value_numeric'].sum()
        transaction_count = len(price_data)
        avg_transaction = price_data['value_numeric'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Chiffre d'Affaires Total",
                value=f"{total_revenue:,.2f}€".replace(',', ' '),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Nombre de Transactions",
                value=transaction_count,
                delta=None
            )
        
        with col3:
            st.metric(
                label="Transaction Moyenne",
                value=f"{avg_transaction:,.2f}€".replace(',', ' '),
                delta=None
            )
        
        # Évolution du CA cumulé
        st.subheader("📈 Évolution du Chiffre d'Affaires Cumulé")
        
        # Grouper les revenus par jour complet pour éviter les points multiples dans la journée
        daily_revenue = price_data.groupby(
            price_data['date'].dt.date
        )['value_numeric'].sum().reset_index()
        daily_revenue.columns = ['date', 'daily_revenue']
        
        # Convertir la date en datetime pour un meilleur affichage
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        
        # Calculer le CA cumulé par jour
        daily_revenue = daily_revenue.sort_values('date')
        daily_revenue['cumulative_revenue'] = daily_revenue['daily_revenue'].cumsum()
        
        fig_revenue = px.line(
            daily_revenue,
            x='date',
            y='cumulative_revenue',
            title="Chiffre d'Affaires Cumulé par Jour",
            labels={'cumulative_revenue': 'CA Cumulé (€)', 'date': 'Date'}
        )
        
        # Ajout de markers pour les points de vente quotidiens
        fig_revenue.add_scatter(
            x=daily_revenue['date'],
            y=daily_revenue['cumulative_revenue'].round(2),
            mode='markers',
            name='CA Quotidien',
            marker=dict(size=8, color='red', symbol='circle'),
            hovertemplate='<b>%{x}</b><br>' +
                         'CA cumulé: €%{y:,.2f}<br>' +
                         '<extra></extra>'
        )
        
        # Configurer l'axe des dates intelligemment
        configure_date_axis(fig_revenue, daily_revenue['date'])
        
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Répartition des prix si plusieurs valeurs
        if transaction_count > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💵 Répartition des Prix")
                
                # Utiliser un histogramme pour mieux visualiser la distribution avec outliers
                fig_price_dist = px.histogram(
                    price_data,
                    x='value_numeric',
                    title="Distribution des Prix",
                    labels={'value_numeric': 'Prix (€)', 'count': 'Nombre de Transactions'},
                    nbins=min(25, len(price_data) // 3)  # Adapter le nombre de bins
                )
                
                # Améliorer la lisibilité
                fig_price_dist.update_layout(
                    height=450,
                    showlegend=False,
                    xaxis=dict(
                        title="Prix (€)",
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)'
                    ),
                    yaxis=dict(
                        title="Nombre de Transactions",
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)'
                    )
                )
                
                st.plotly_chart(fig_price_dist, use_container_width=True)
            
            with col2:
                st.subheader("📦 CA par Type d'Événement")
                revenue_by_event = price_data.groupby('name')['value_numeric'].sum()
                
                fig_revenue_by_event = px.pie(
                    values=revenue_by_event.values,
                    names=revenue_by_event.index,
                    title="Répartition du CA"
                )
                st.plotly_chart(fig_revenue_by_event, use_container_width=True)
    
    # === ANALYSE GÉNÉRIQUE DES AUTRES CLÉS-VALEURS ===
    st.markdown("---")
    st.subheader("📋 Analyse des Métadonnées")
    
    # Identifier toutes les clés-valeurs non-prix
    all_value_keys = filtered_data['value_key'].unique()
    non_price_keys = [
        key for key in all_value_keys 
        if not ('price' in key.lower() or 'prix' in key.lower()) and key != 'no_value'
    ]
    
    if non_price_keys:
        st.write(f"**Analyse détaillée des {len(non_price_keys)} types de métadonnées détectées**")
        
        # Créer des onglets ou colonnes selon le nombre de clés
        if len(non_price_keys) <= 3:
            # Si 3 clés ou moins, utiliser des colonnes
            cols = st.columns(len(non_price_keys))
            
            for idx, value_key in enumerate(sorted(non_price_keys)):
                with cols[idx]:
                    display_value_key_analysis(filtered_data, value_key)
        else:
            # Si plus de 3 clés, utiliser des onglets
            tabs = st.tabs([f"📊 {key.title()}" for key in sorted(non_price_keys)])
            
            for idx, value_key in enumerate(sorted(non_price_keys)):
                with tabs[idx]:
                    display_value_key_analysis(filtered_data, value_key)
    else:
        st.info("Aucune métadonnée non-prix détectée dans les événements sélectionnés")
    
    # === INFORMATIONS COMPLÉMENTAIRES ===
    with st.expander("🔍 Détails des événements sélectionnés"):
        st.write("**Répartition par événement:**")
        # Utiliser unique_events pour compter correctement
        event_counts = unique_events['name'].value_counts()
        for event_name, count in event_counts.items():
            percentage = (count / total_events) * 100
            st.write(f"- **{event_name}**: {count} événements ({percentage:.1f}%)")
        
        st.write("**Types de valeurs présentes:**")
        value_types = filtered_data['value_key'].value_counts()
        for value_type, count in value_types.items():
            st.write(f"- **{value_type}**: {count} occurrences")
    
    # === DONNÉES COMPLÈTES FILTRÉES ===
    st.markdown("---")
    st.subheader("📋 Données complètes des événements sélectionnés")
    
    # Utiliser les données consolidées pour l'affichage
    if hasattr(processor, 'consolidated_data') and processor.consolidated_data is not None:
        # Filtrer les données consolidées par nom d'événement
        consolidated_filtered = processor.consolidated_data[
            processor.consolidated_data['name'].isin(selected_event_names)
        ].copy()
        
        if not consolidated_filtered.empty:
            # Formatage des dates pour un meilleur affichage
            consolidated_filtered['date'] = pd.to_datetime(consolidated_filtered['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Tri par date (plus récent en haut)
            consolidated_filtered = consolidated_filtered.sort_values('date', ascending=False)
            
            # Réorganisation des colonnes : colonnes de base puis colonnes de valeurs
            base_columns = ['name', 'date', 'conversation_id']
            value_columns = [col for col in consolidated_filtered.columns if col not in base_columns]
            
            # Réorganiser les colonnes de valeurs par paires (text, numeric)
            organized_columns = base_columns.copy()
            
            # Grouper les colonnes par type de valeur
            value_types = set()
            for col in value_columns:
                if col.endswith('_text') or col.endswith('_numeric'):
                    value_type = col.replace('_text', '').replace('_numeric', '')
                    value_types.add(value_type)
            
            # Ajouter les colonnes dans l'ordre : text puis numeric pour chaque type
            for value_type in sorted(value_types):
                text_col = f'{value_type}_text'
                numeric_col = f'{value_type}_numeric'
                if text_col in consolidated_filtered.columns:
                    organized_columns.append(text_col)
                if numeric_col in consolidated_filtered.columns:
                    organized_columns.append(numeric_col)
            
            display_data = consolidated_filtered[organized_columns]
            
            # Renommage des colonnes pour plus de clarté
            column_rename = {
                'name': 'Nom Événement',
                'date': 'Date',
                'conversation_id': 'ID Conversation'
            }
            
            # Renommer les colonnes de valeurs
            for col in display_data.columns:
                if col.endswith('_text'):
                    value_type = col.replace('_text', '')
                    column_rename[col] = f'{value_type.title()}'
                elif col.endswith('_numeric'):
                    value_type = col.replace('_numeric', '')
                    column_rename[col] = f'{value_type.title()} (€)'
            
            display_data = display_data.rename(columns=column_rename)
            
            # Affichage du DataFrame avec des options d'interaction
            st.write(f"**{len(display_data)} événements uniques affichés**")
            
            # Options d'affichage
            col1, col2 = st.columns(2)
            
            with col1:
                show_all = st.checkbox("Afficher toutes les lignes", key="show_all_data")
            
            with col2:
                if st.button("📥 Télécharger les données filtrées", key="download_filtered"):
                    csv = display_data.to_csv(index=False)
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name=f"donnees_filtrees_{len(selected_event_names)}_evenements.csv",
                        mime="text/csv"
                    )
            
            # Affichage du DataFrame
            if show_all:
                st.dataframe(
                    display_data,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                # Affichage limité aux 50 premières lignes
                st.dataframe(
                    display_data.head(50),
                    use_container_width=True,
                    hide_index=True
                )
                if len(display_data) > 50:
                    st.info(f"Affichage limité aux 50 premières lignes. Cochez 'Afficher toutes les lignes' pour voir les {len(display_data)} lignes complètes.")
        else:
            st.warning("Aucune donnée consolidée trouvée pour les événements sélectionnés")
    else:
        st.error("Données consolidées non disponibles")


def display_value_key_analysis(filtered_data: pd.DataFrame, value_key: str):
    """Affiche l'analyse détaillée d'une clé-valeur spécifique"""
    # Filtrer les données pour cette clé-valeur
    key_data = filtered_data[filtered_data['value_key'] == value_key].copy()
    
    if key_data.empty:
        st.warning(f"Aucune donnée trouvée pour la clé '{value_key}'")
        return
    
    # Compter les occurrences de chaque valeur
    value_counts = key_data['value_text'].value_counts()
    total_count = len(key_data)
    
    # Calculer les prix associés pour chaque valeur de métadonnée
    # On va chercher les prix dans les événements de même conversation_id, name et date
    price_by_metadata = {}
    has_associated_prices = False
    
    for value in value_counts.index:
        # Récupérer les événements avec cette valeur de métadonnée
        metadata_events = key_data[key_data['value_text'] == value]
        total_revenue = 0
        
        # Pour chaque événement avec cette métadonnée, chercher les prix associés
        for _, event in metadata_events.iterrows():
            # Chercher les prix dans la même conversation/date/nom
            matching_prices = filtered_data[
                (filtered_data['name'] == event['name']) &
                (filtered_data['date'] == event['date']) &
                (filtered_data['conversation_id'] == event['conversation_id']) &
                (filtered_data['value_key'].str.contains('price', case=False, na=False)) &
                (filtered_data['value_numeric'].notna())
            ]
            
            if not matching_prices.empty:
                total_revenue += matching_prices['value_numeric'].sum()
                has_associated_prices = True
        
        price_by_metadata[value] = total_revenue
    
    # Créer le DataFrame avec occurrences, pourcentages et prix
    analysis_data = []
    total_revenue_all = sum(price_by_metadata.values()) if has_associated_prices else 0
    
    for value, count in value_counts.items():
        percentage = (count / total_count) * 100
        revenue = price_by_metadata.get(value, 0)
        
        row_data = {
            'Valeur': value if value else '(vide)',
            'Occurrences': count,
            'Pourcentage': f"{percentage:.1f}%"
        }
        
        # Ajouter les colonnes de prix si applicable
        if has_associated_prices:
            row_data['CA Associé (€)'] = f"{revenue:,.2f}".replace(',', ' ')
            if total_revenue_all > 0:
                revenue_percentage = (revenue / total_revenue_all) * 100
                row_data['% du CA'] = f"{revenue_percentage:.1f}%"
        
        analysis_data.append(row_data)
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Affichage du titre et des métriques
    st.write(f"**{value_key.title().replace('_', ' ')}**")
    
    # Métriques principales
    if has_associated_prices:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Valeurs uniques", len(value_counts))
        with col2:
            st.metric("Total occurrences", total_count)
        with col3:
            st.metric("CA Total", f"{total_revenue_all:,.2f}€".replace(',', ' '))
        with col4:
            avg_revenue = total_revenue_all / total_count if total_count > 0 else 0
            st.metric("CA Moyen", f"{avg_revenue:,.2f}€".replace(',', ' '))
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Valeurs uniques", len(value_counts))
        with col2:
            st.metric("Total occurrences", total_count)
    
    # Tableau des résultats
    st.dataframe(
        analysis_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Graphique si plus d'une valeur
    if len(value_counts) > 1:
        if has_associated_prices and total_revenue_all > 0:
            # Double graphique : occurrences et CA
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart pour les occurrences
                fig_occurrences = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Répartition par Occurrences",
                    height=300
                )
                fig_occurrences.update_traces(textposition='inside', textinfo='percent+label')
                fig_occurrences.update_layout(showlegend=False)
                st.plotly_chart(fig_occurrences, use_container_width=True)
            
            with col2:
                # Pie chart pour le CA
                revenue_values = [price_by_metadata.get(name, 0) for name in value_counts.index]
                fig_revenue = px.pie(
                    values=revenue_values,
                    names=value_counts.index,
                    title=f"Répartition par CA (€)",
                    height=300
                )
                fig_revenue.update_traces(textposition='inside', textinfo='percent+label')
                fig_revenue.update_layout(showlegend=False)
                st.plotly_chart(fig_revenue, use_container_width=True)
        else:
            # Pie chart simple pour la répartition
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Répartition - {value_key.title().replace('_', ' ')}",
                height=300
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Top des valeurs par CA si applicable
    if has_associated_prices and total_revenue_all > 0 and len(value_counts) > 1:
        st.write("💰 **Top par Chiffre d'Affaires:**")
        
        # Créer un ranking par CA
        revenue_ranking = sorted(
            [(value, price_by_metadata.get(value, 0)) for value in value_counts.index],
            key=lambda x: x[1],
            reverse=True
        )
        
        ranking_col1, ranking_col2 = st.columns(2)
        
        with ranking_col1:
            st.write("**🥇 Meilleur CA:**")
            best_value, best_revenue = revenue_ranking[0]
            best_count = value_counts[best_value]
            st.write(f"• **{best_value}**: {best_revenue:,.2f}€ ({best_count} occurrences)".replace(',', ' '))
        
        with ranking_col2:
            if len(revenue_ranking) > 1:
                st.write("**📈 Panier moyen par valeur:**")
                for value, revenue in revenue_ranking:
                    count = value_counts[value]
                    avg_basket = revenue / count if count > 0 else 0
                    st.write(f"• **{value}**: {avg_basket:,.2f}€/transaction".replace(',', ' '))
    
    # Informations complémentaires si valeurs numériques détectées
    numeric_data = key_data.dropna(subset=['value_numeric'])
    if not numeric_data.empty:
        st.write("📊 **Statistiques numériques détectées:**")
        numeric_col1, numeric_col2, numeric_col3 = st.columns(3)
        
        with numeric_col1:
            st.metric("Minimum", f"{numeric_data['value_numeric'].min():.2f}")
        with numeric_col2:
            st.metric("Moyenne", f"{numeric_data['value_numeric'].mean():.2f}")
        with numeric_col3:
            st.metric("Maximum", f"{numeric_data['value_numeric'].max():.2f}")


def display_detailed_analysis(processor: ROIDataProcessor, selected_events: List[str]):
    """Affiche l'analyse détaillée des événements sélectionnés"""
    st.markdown('<div class="section-header">📈 Analyse Détaillée</div>', 
                unsafe_allow_html=True)
    
    analysis_results = processor.analyze_selected_events(selected_events)
    
    if 'error' in analysis_results:
        st.error(analysis_results['error'])
        return
    
    # Calcul des totaux pour le mode revenus
    total_revenue = 0
    total_transactions = 0
    revenue_events = []
    
    for event, analysis in analysis_results.items():
        if analysis.get('type') == 'price' and 'total_revenue' in analysis:
            total_revenue += analysis['total_revenue']
            total_transactions += analysis['transaction_count']
            revenue_events.append(event)
    
    # Affichage des métriques principales si mode revenus activé
    if total_revenue > 0:
        st.subheader("💰 Mode Chiffre d'Affaires Activé")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Chiffre d'Affaires Total",
                value=f"{total_revenue:,.2f}€".replace(',', ' '),
                delta=None
            )
        with col2:
            st.metric(
                label="Nombre de Transactions",
                value=total_transactions,
                delta=None
            )
        with col3:
            avg_transaction = total_revenue / total_transactions if total_transactions > 0 else 0
            st.metric(
                label="Transaction Moyenne",
                value=f"{avg_transaction:,.2f}€".replace(',', ' '),
                delta=None
            )
    
    # Analyse détaillée par événement
    for event, analysis in analysis_results.items():
        with st.expander(f"📊 Analyse de '{event}' (Type: {analysis.get('type', 'unknown')})"):
            
            if analysis.get('type') == 'price':
                display_price_analysis(event, analysis)
            elif analysis.get('type') == 'product':
                display_product_analysis(event, analysis)
            else:
                display_custom_analysis(event, analysis)


def display_price_analysis(event: str, analysis: Dict):
    """Affiche l'analyse spécialisée pour les prix"""
    if 'error' in analysis:
        st.error(analysis['error'])
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Répartition des prix**")
        if 'distribution' in analysis and 'temporal_data' in analysis:
            # Utiliser les données originales pour l'histogramme
            temporal_data = analysis['temporal_data']
            
            # Vérifier s'il y a une colonne avec les prix individuels
            if 'daily_revenue' in temporal_data.columns:
                # Si on a les données agrégées par jour, on ne peut pas faire un histogramme détaillé
                # On utilisera alors la distribution existante
                price_dist_df = pd.DataFrame({
                    'prix': analysis['distribution'].index,
                    'frequence': analysis['distribution'].values
                }).sort_values('prix')
                
                fig_price_dist = px.histogram(
                    x=price_dist_df['prix'].repeat(price_dist_df['frequence']),
                    title=f"Distribution des prix - {event}",
                    labels={'x': 'Prix (€)', 'count': 'Fréquence'},
                    nbins=min(20, len(price_dist_df))
                )
            else:
                # Utiliser directement les données si disponibles
                fig_price_dist = px.histogram(
                    analysis['distribution'].index.repeat(analysis['distribution'].values),
                    title=f"Distribution des prix - {event}",
                    labels={'x': 'Prix (€)', 'count': 'Fréquence'},
                    nbins=min(20, len(analysis['distribution']))
                )
            
            # Améliorer la lisibilité
            fig_price_dist.update_layout(
                height=400,
                showlegend=False,
                xaxis=dict(
                    title="Prix (€)",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    title="Fréquence",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                )
            )
            
            st.plotly_chart(fig_price_dist, use_container_width=True)
    
    with col2:
        st.write("**Évolution du CA cumulé**")
        if 'temporal_data' in analysis:
            temporal_data = analysis['temporal_data']
            fig_cumul = px.line(
                temporal_data,
                x='date',
                y='cumulative_revenue',
                title=f"CA cumulé - {event}",
                labels={'cumulative_revenue': 'CA Cumulé (€)', 'date': 'Date'}
            )
            # Configurer l'axe des dates intelligemment
            configure_date_axis(fig_cumul, temporal_data['date'])
            st.plotly_chart(fig_cumul, use_container_width=True)


def display_product_analysis(event: str, analysis: Dict):
    """Affiche l'analyse spécialisée pour les produits"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Répartition des produits**")
        if 'distribution' in analysis:
            fig_product_dist = px.pie(
                values=analysis['distribution'].values,
                names=analysis['distribution'].index,
                title=f"Répartition des produits - {event}"
            )
            fig_product_dist.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_product_dist, use_container_width=True)
    
    with col2:
        st.write("**Top produits**")
        if 'top_products' in analysis:
            top_products = analysis['top_products']
            fig_top = px.bar(
                x=top_products.values,
                y=top_products.index,
                orientation='h',
                title=f"Top produits - {event}",
                labels={'x': 'Nombre de ventes', 'y': 'Produit'}
            )
            st.plotly_chart(fig_top, use_container_width=True)


def display_custom_analysis(event: str, analysis: Dict):
    """Affiche l'analyse pour les données personnalisées"""
    st.write("**Répartition des valeurs**")
    if 'distribution' in analysis:
        fig_custom = px.bar(
            x=analysis['distribution'].values,
            y=analysis['distribution'].index,
            orientation='h',
            title=f"Distribution - {event}",
            labels={'x': 'Fréquence', 'y': 'Valeur'}
        )
        st.plotly_chart(fig_custom, use_container_width=True)


def analyze_conversion_performance(processor: ROIDataProcessor, conversion_events, total_genii_conversations=None) -> Dict:
    """Analyse complète de la performance des événements de conversion sélectionnés"""
    
    # Gestion de liste ou d'un seul événement
    events_list = conversion_events if isinstance(conversion_events, list) else [conversion_events]
    
    active_data = processor.get_active_data()
    
    # Conversion des dates
    active_data['date'] = pd.to_datetime(active_data['date'], errors='coerce')
    if active_data['date'].dt.tz is not None:
        active_data['date'] = active_data['date'].dt.tz_convert(None)
    
    # 1. Analyser tous les événements de conversion sélectionnés
    conversion_events_data = active_data[active_data['name'].isin(events_list)].copy()
    
    if conversion_events_data.empty:
        events_str = "', '".join(events_list)
        return {'error': f"Aucun événement trouvé parmi '{events_str}'"}
    
    # 2. Détecter s'il y a des prix directement dans les événements de conversion
    conversion_with_prices = conversion_events_data[
        (conversion_events_data['value_key'].str.contains('price', case=False, na=False) | 
         conversion_events_data['value_key'].str.contains('prix', case=False, na=False)) &
        (conversion_events_data['value_numeric'].notna())
    ]
    
    # 3. Compter les conversations uniques
    unique_conversations = conversion_events_data['conversation_id'].unique()
    valid_conversations = [
        conv_id for conv_id in unique_conversations 
        if not (pd.isna(conv_id) or conv_id == '')
    ]
    
    result = {
        'conversion_events': events_list,
        'total_conversion_events': len(conversion_events_data),
        'unique_conversations': len(valid_conversations),
        'has_direct_prices': not conversion_with_prices.empty,
        'analysis_mode': 'direct_prices' if not conversion_with_prices.empty else 'price_tracking'
    }
    
    # 4. Calcul du taux de conversion si total Genii fourni
    if total_genii_conversations and total_genii_conversations > 0:
        result['conversion_rate'] = (len(valid_conversations) / total_genii_conversations) * 100
    
    # 5. Analyse selon le mode détecté
    if result['analysis_mode'] == 'direct_prices':
        # Mode prix directs
        total_revenue = conversion_with_prices['value_numeric'].sum()
        avg_price = conversion_with_prices['value_numeric'].mean()
        transaction_count = len(conversion_with_prices)
        
        # Évolution temporelle
        daily_revenue = conversion_with_prices.groupby(
            conversion_with_prices['date'].dt.date
        )['value_numeric'].sum().reset_index()
        daily_revenue.columns = ['date', 'daily_revenue']
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        daily_revenue = daily_revenue.sort_values('date')
        daily_revenue['cumulative_revenue'] = daily_revenue['daily_revenue'].cumsum()
        
        result.update({
            'total_revenue': total_revenue,
            'average_price': avg_price,
            'transaction_count': transaction_count,
            'temporal_data': daily_revenue
        })
        
    else:
        # Mode tracking du dernier prix
        tracking_result = analyze_conversion_tracking(processor, events_list)
        if 'error' not in tracking_result:
            # Créer les données temporelles à partir des résultats de tracking
            temporal_data = tracking_result['analysis_data'].groupby(
                tracking_result['analysis_data']['conversion_date'].dt.date
            ).agg({
                'last_price': 'sum',
                'conversation_id': 'count'
            }).reset_index()
            temporal_data.columns = ['date', 'daily_revenue', 'daily_conversions']
            
            # Calcul cumulatif
            temporal_data['date'] = pd.to_datetime(temporal_data['date'])
            temporal_data = temporal_data.sort_values('date')
            temporal_data['cumulative_revenue'] = temporal_data['daily_revenue'].cumsum()
            
            result.update({
                'total_revenue': tracking_result['total_revenue'],
                'average_price': tracking_result['average_price'], 
                'successful_conversions': tracking_result['successful_conversions'],
                'temporal_data': temporal_data
            })
        else:
            result.update({
                'total_revenue': 0,
                'average_price': 0,
                'successful_conversions': 0,
                'tracking_error': tracking_result['error']
            })
    
    return result


def analyze_conversion_tracking(processor: ROIDataProcessor, conversion_events) -> Dict:
    """Analyse le tracking de conversion en trouvant le dernier prix avant les événements de conversion"""
    
    # Gestion de liste ou d'un seul événement
    events_list = conversion_events if isinstance(conversion_events, list) else [conversion_events]
    
    # Utiliser les données actives (filtrées ou complètes)
    active_data = processor.get_active_data()
    
    # Conversion des dates
    active_data['date'] = pd.to_datetime(active_data['date'], errors='coerce')
    
    # Normaliser les timezones pour éviter les erreurs de comparaison
    if active_data['date'].dt.tz is not None:
        active_data['date'] = active_data['date'].dt.tz_convert(None)
    
    # 1. Trouver toutes les conversations qui contiennent au moins un des événements de conversion
    conversations_with_conversion = active_data[
        active_data['name'].isin(events_list)
    ]['conversation_id'].unique()
    
    if len(conversations_with_conversion) == 0:
        events_str = "', '".join(events_list)
        return {'error': f"Aucune conversation trouvée avec les événements '{events_str}'"}
    
    conversion_analysis = []
    total_revenue = 0
    
    for conv_id in conversations_with_conversion:
        if pd.isna(conv_id) or conv_id == '':
            continue
            
        # 2. Récupérer tous les événements de cette conversation, triés par date
        conv_events = active_data[
            active_data['conversation_id'] == conv_id
        ].sort_values('date')
        
        # 3. Trouver la première occurrence d'un des événements de conversion
        conversion_events_in_conv = conv_events[conv_events['name'].isin(events_list)]
        if conversion_events_in_conv.empty:
            continue
            
        first_conversion = conversion_events_in_conv.iloc[0]
        first_conversion_date = first_conversion['date']
        first_conversion_event = first_conversion['name']
        
        # 4. Chercher tous les événements avec prix avant cette date de conversion
        price_events_before = conv_events[
            (conv_events['date'] < first_conversion_date) &
            (conv_events['value_key'].str.contains('price', case=False, na=False) | 
             conv_events['value_key'].str.contains('prix', case=False, na=False)) &
            (conv_events['value_numeric'].notna())
        ]
        
        # 5. Prendre le dernier prix (le plus proche de la conversion)
        if not price_events_before.empty:
            last_price_event = price_events_before.iloc[-1]  # Le dernier chronologiquement
            last_price = last_price_event['value_numeric']
            
            conversion_analysis.append({
                'conversation_id': conv_id,
                'conversion_date': first_conversion_date,
                'conversion_event': first_conversion_event,
                'last_price': last_price,
                'last_price_date': last_price_event['date'],
                'last_price_event': last_price_event['name'],
                'time_to_conversion': (first_conversion_date - last_price_event['date']).total_seconds() / 60  # en minutes
            })
            
            total_revenue += last_price
    
    # Création du DataFrame pour l'analyse
    if not conversion_analysis:
        events_str = "', '".join(events_list)
        return {'error': f"Aucun prix trouvé avant les événements de conversion '{events_str}'"}
    
    analysis_df = pd.DataFrame(conversion_analysis)
    
    # Tri par date de conversion pour l'évolution cumulative
    analysis_df = analysis_df.sort_values('conversion_date')
    analysis_df['cumulative_revenue'] = analysis_df['last_price'].cumsum()
    
    return {
        'success': True,
        'conversion_events': events_list,
        'total_conversations': len(conversations_with_conversion),
        'successful_conversions': len(analysis_df),
        'total_revenue': total_revenue,
        'average_price': analysis_df['last_price'].mean(),
        'analysis_data': analysis_df,
        'price_distribution': analysis_df['last_price'].value_counts().sort_index(),
        'avg_time_to_conversion': analysis_df['time_to_conversion'].mean()
    }


def display_conversion_performance_section(processor: ROIDataProcessor, conversion_events, total_genii_conversations=None):
    """Affiche la section de performance de conversion (prioritaire après configuration)"""
    
    if not conversion_events:
        return
    
    # Gestion de liste ou d'un seul événement
    events_list = conversion_events if isinstance(conversion_events, list) else [conversion_events]
    
    st.markdown('<div class="section-header">🎯 Indicateurs de Performance de Conversion</div>', 
                unsafe_allow_html=True)
    
    with st.spinner("Analyse des performances de conversion..."):
        performance = analyze_conversion_performance(processor, events_list, total_genii_conversations)
    
    if 'error' in performance:
        st.error(f"❌ {performance['error']}")
        return
    
    # === BADGE MODE D'ANALYSE ===
    mode_text = "📊 Prix Directs" if performance['analysis_mode'] == 'direct_prices' else "🔍 Tracking du Dernier Prix"
    events_text = ", ".join([f"'{e}'" for e in events_list])
    st.info(f"**Mode d'analyse détecté** : {mode_text} | **Événements analysés** : {events_text}")
    
    # === MÉTRIQUES PRINCIPALES ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Conversions",
            value=f"{performance['total_conversion_events']:,}",
            delta=f"sur {performance['unique_conversations']} conversations uniques" if performance['total_conversion_events'] > performance['unique_conversations'] else f"{performance['unique_conversations']} conversations uniques"
        )
    
    with col2:
        if 'conversion_rate' in performance:
            # Calculer le taux basé sur les événements totaux
            events_conversion_rate = (performance['total_conversion_events'] / total_genii_conversations) * 100
            st.metric(
                label="Taux de Conversion",
                value=f"{events_conversion_rate:.2f}%", 
                delta=f"sur {total_genii_conversations} conversations Genii"
            )
        else:
            st.metric(
                label="Taux de Conversion",
                value="N/A",
                delta="Total conversations Genii non spécifié"
            )
    
    with col3:
        if performance.get('total_revenue', 0) > 0:
            st.metric(
                label="Chiffre d'Affaires Total",
                value=f"{performance['total_revenue']:,.2f}€".replace(',', ' '),
                delta=None
            )
        else:
            if 'tracking_error' in performance:
                st.metric(
                    label="Chiffre d'Affaires",
                    value="Non disponible",
                    delta="Aucun prix identifiable"
                )
            else:
                st.metric(
                    label="Chiffre d'Affaires Total",
                    value="0€",
                    delta=None
                )
    
    with col4:
        if performance.get('average_price', 0) > 0:
            st.metric(
                label="Panier Moyen",
                value=f"{performance['average_price']:,.2f}€".replace(',', ' '),
                delta=None
            )
        else:
            st.metric(
                label="Panier Moyen",
                value="N/A",
                delta=None
            )
    
    # === GRAPHIQUE D'ÉVOLUTION ===
    if 'temporal_data' in performance and not performance['temporal_data'].empty:
        temporal_data = performance['temporal_data']
        
        # Graphique principal : évolution cumulative
        fig_evolution = go.Figure()
        
        # Ligne cumulative
        fig_evolution.add_trace(go.Scatter(
            x=temporal_data['date'],
            y=temporal_data['cumulative_revenue'].round(2),
            mode='lines+markers',
            name='CA Cumulé',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#1f77b4'),
            hovertemplate='<b>%{x}</b><br>' +
                         'CA cumulé: €%{y:,.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Points quotidiens
        fig_evolution.add_trace(go.Scatter(
            x=temporal_data['date'],
            y=temporal_data['daily_revenue'].round(2),
            mode='markers',
            name='CA quotidien',
            marker=dict(size=10, color='red', symbol='diamond'),
            hovertemplate='<b>%{x}</b><br>' +
                         'CA du jour: €%{y:,.2f}<br>' +
                         '<extra></extra>'
        ))
        
        events_title = ", ".join(events_list)
        fig_evolution.update_layout(
            title=f"Performance de Conversion - {events_title}",
            xaxis_title="Date",
            yaxis_title="Montant (€)",
            hovermode='x unified',
            height=400
        )
        
        # Configurer l'axe des dates intelligemment
        configure_date_axis(fig_evolution, temporal_data['date'])
        
        st.plotly_chart(fig_evolution, use_container_width=True)
    
    # === INFORMATIONS COMPLÉMENTAIRES ===
    if performance['analysis_mode'] == 'price_tracking' and 'tracking_error' in performance:
        st.warning(f"⚠️ {performance['tracking_error']}")
        st.info("💡 Cet événement de conversion ne contient pas de prix et aucun prix n'a pu être identifié dans les événements précédents des conversations.")


def display_conversion_analysis(processor: ROIDataProcessor):
    """Affiche l'interface d'analyse de conversion (Niveau 1.5 - entre général et spécialisé)"""
    st.markdown('<div class="section-header">🎯 Analyse de Conversion - Tracking du Dernier Prix</div>', 
                unsafe_allow_html=True)
    
    # Utiliser les données actives (filtrées ou complètes)
    active_data = processor.get_active_data()
    
    # Menu déroulant pour sélectionner l'événement de conversion
    available_events = sorted(active_data['name'].unique())
    
    if not available_events:
        st.warning("Aucun événement trouvé dans les données")
        return
    
    st.subheader("🎯 Sélection de l'événement de conversion finale")
    st.write("Sélectionnez l'événement qui marque la conversion réussie (ex: 'Step 4, Conversion', 'Purchase Complete', etc.)")
    
    selected_conversion = st.selectbox(
        "Événement de conversion :",
        options=["Sélectionnez un événement..."] + available_events,
        index=0,
        help="L'événement qui confirme qu'une transaction/conversion a été réalisée"
    )
    
    
    if selected_conversion and selected_conversion != "Sélectionnez un événement...":
        # Analyse automatique
        with st.spinner("Analyse des conversions en cours..."):
            analysis_result = analyze_conversion_tracking(processor, selected_conversion)
        
        if 'error' in analysis_result:
            st.error(f"❌ {analysis_result['error']}")
            return
        
        # === MÉTRIQUES PRINCIPALES ===
        st.subheader("📊 Résultats de l'analyse de conversion")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Conversions avec Prix Identifiable",
                value=f"{analysis_result['successful_conversions']:,}",
                delta=f"sur {analysis_result['total_conversations']} conversations"
            )
        
        with col2:
            st.metric(
                label="Chiffre d'Affaires Total",
                value=f"{analysis_result['total_revenue']:,.2f}€".replace(',', ' '),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Panier Moyen",
                value=f"{analysis_result['average_price']:,.2f}€".replace(',', ' '),
                delta=None
            )
        
        # === GRAPHIQUE D'ÉVOLUTION CUMULATIVE ===
        st.subheader("📈 Évolution du Chiffre d'Affaires Cumulé")
        
        analysis_df = analysis_result['analysis_data']
        
        # Grouper les conversions par jour complet pour éviter les points multiples dans la journée
        daily_conversions = analysis_df.copy()
        daily_conversions['conversion_date_day'] = daily_conversions['conversion_date'].dt.date
        
        # Agréger par jour : sommer les prix et compter les conversions
        daily_revenue = daily_conversions.groupby('conversion_date_day').agg({
            'last_price': 'sum',  # Somme des revenus du jour
            'conversation_id': 'count'  # Nombre de conversions du jour
        }).reset_index()
        daily_revenue.columns = ['date', 'daily_revenue', 'daily_conversions']
        
        # Convertir la date en datetime pour un meilleur affichage
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        
        # Calculer le CA cumulé par jour
        daily_revenue = daily_revenue.sort_values('date')
        daily_revenue['cumulative_revenue'] = daily_revenue['daily_revenue'].cumsum()
        
        # Graphique principal : évolution cumulative
        fig_cumulative = go.Figure()
        
        # Ligne cumulative
        fig_cumulative.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['cumulative_revenue'].round(2),
            mode='lines+markers',
            name='CA Cumulé',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#1f77b4'),
            hovertemplate='<b>%{x}</b><br>' +
                         'CA cumulé: €%{y:,.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Ajout des points de conversion quotidiens
        fig_cumulative.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['daily_revenue'].round(2),
            mode='markers',
            name='CA quotidien',
            marker=dict(size=10, color='red', symbol='diamond'),
            hovertemplate='<b>%{x}</b><br>' +
                         'CA du jour: €%{y:,.2f}<br>' +
                         'Conversions: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=daily_revenue['daily_conversions']
        ))
        
        fig_cumulative.update_layout(
            title=f"Évolution du CA pour les conversions '{selected_conversion}' (par jour)",
            xaxis_title="Date de conversion",
            yaxis_title="Montant (€)",
            hovermode='x unified',
            height=500
        )
        
        # Configurer l'axe des dates intelligemment
        configure_date_axis(fig_cumulative, daily_revenue['date'])
        
        st.plotly_chart(fig_cumulative, use_container_width=True)
        
        # === GRAPHIQUES SECONDAIRES ===
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Distribution des Prix")
            
            if len(analysis_result['price_distribution']) > 1:
                analysis_df = analysis_result['analysis_data']
                
                # Créer l'histogramme
                fig_price_dist = px.histogram(
                    analysis_df,
                    x='last_price',
                    title="Distribution des derniers prix avant conversion",
                    labels={'last_price': 'Prix (€)', 'count': 'Nombre de conversions'},
                    nbins=min(30, len(analysis_df) // 2)  # Adapter le nombre de bins
                )
                
                # Améliorer la lisibilité
                fig_price_dist.update_layout(
                    height=500,
                    showlegend=False,
                    xaxis=dict(
                        title="Prix (€)",
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)'
                    ),
                    yaxis=dict(
                        title="Nombre de conversions",
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)'
                    )
                )
                
                st.plotly_chart(fig_price_dist, use_container_width=True)
                    
            else:
                st.info("Tous les prix de conversion sont identiques")
        
        with col2:
            st.subheader("⏱️ Temps jusqu'à conversion")
            
            if analysis_result['avg_time_to_conversion'] > 0:
                # Histogramme du temps jusqu'à conversion
                fig_time_dist = px.histogram(
                    analysis_df,
                    x='time_to_conversion',
                    title="Temps entre dernier prix et conversion",
                    labels={'time_to_conversion': 'Minutes', 'count': 'Nombre de conversions'},
                    nbins=20
                )
                fig_time_dist.update_layout(height=400)
                st.plotly_chart(fig_time_dist, use_container_width=True)
                
                # Métrique temps moyen
                st.metric(
                    label="Temps moyen jusqu'à conversion",
                    value=f"{analysis_result['avg_time_to_conversion']:.1f}min"
                )
            else:
                st.info("Temps de conversion instantané")
        
        # === DÉTAILS COMPLÉMENTAIRES ===
        with st.expander(f"🔍 Détails des conversions pour '{selected_conversion}'"):
            st.write("**Résumé de l'analyse :**")
            st.write(f"• **{analysis_result['total_conversations']}** conversations uniques contiennent l'événement '{selected_conversion}'")
            st.write(f"• **{analysis_result['successful_conversions']}** de ces conversations ont un prix identifiable avant la conversion")
            
            # Calcul du nombre d'événements totaux pour comparaison
            active_data = processor.get_active_data()
            total_events = len(active_data[active_data['name'] == selected_conversion])
            
            if total_events > analysis_result['total_conversations']:
                st.write(f"• **{total_events}** événements de conversion au total (certaines conversations en ont plusieurs)")
            
            excluded_conversations = analysis_result['total_conversations'] - analysis_result['successful_conversions']
            if excluded_conversations > 0:
                st.write(f"• **{excluded_conversations}** conversations exclues (pas de prix identifiable ou ID conversation manquant)")
                
            st.write(f"• **Chiffre d'affaires total** : {analysis_result['total_revenue']:,.2f}€".replace(',', ' '))
            st.write(f"• **Prix moyen** : {analysis_result['average_price']:,.2f}€".replace(',', ' '))
            
            # Tableau détaillé des conversions
            st.write("**📋 Détail des conversions :**")
            
            # Formatage du DataFrame pour affichage
            display_df = analysis_df.copy()
            display_df['conversion_date'] = display_df['conversion_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['last_price_date'] = display_df['last_price_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['time_to_conversion'] = display_df['time_to_conversion'].round(2)
            
            # Renommage des colonnes
            display_df = display_df.rename(columns={
                'conversation_id': 'ID Conversation',
                'conversion_date': 'Date Conversion',
                'last_price': 'Dernier Prix (€)',
                'last_price_date': 'Date Dernier Prix',
                'last_price_event': 'Événement du Prix',
                'time_to_conversion': 'Temps jusqu\'à Conversion (min)',
                'cumulative_revenue': 'CA Cumulé (€)'
            })
            
            # Réorganisation des colonnes
            column_order = ['ID Conversation', 'Dernier Prix (€)', 'Date Dernier Prix', 
                          'Événement du Prix', 'Date Conversion', 'Temps jusqu\'à Conversion (min)', 'CA Cumulé (€)']
            display_df = display_df[column_order]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Option de téléchargement
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les détails",
                data=csv,
                file_name=f"conversion_analysis_{selected_conversion.replace(' ', '_')}.csv",
                mime="text/csv"
            )


def main():
    """Fonction principale de l'application"""
    
    # Initialisation des variables de session
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    
    # Titre principal
    st.title("📊 ROI Tracker")
    
    # Zone d'upload
    st.subheader("📁 Upload de fichier")
    uploaded_file = st.file_uploader(
        "Glissez-déposez votre fichier CSV ou cliquez pour le sélectionner",
        type=['csv'],
        help="Le fichier doit contenir: name, date + soit 'values' (format clé:valeur) soit des colonnes 'value_*'"
    )
    
    if uploaded_file is not None:
        # Chargement et validation des données
        processor, message = load_and_validate_data(uploaded_file)
        
        if processor is not None:
            st.success(message)
            st.session_state.processor = processor
            
            # Sélection de la période directement sous l'upload
            start_date, end_date = display_period_picker(processor)
            
            if start_date is not None and end_date is not None:
                # Appliquer le filtre de date
                processor.apply_date_filter(start_date, end_date)
                
                # === CONFIGURATION GENII ===
                col1, col2 = st.columns(2)
                
                with col1:
                    # Champ pour le nombre total de conversations Genii
                    total_genii_conversations = st.number_input(
                        "Nombre total de conversations Genii sur la période",
                        min_value=0,
                        value=st.session_state.get('total_genii_conversations', 0),
                        step=1,
                        help="Saisissez le nombre total de conversations sur votre système Genii pour la période sélectionnée"
                    )
                    
                    # Sauvegarder dans la session
                    st.session_state['total_genii_conversations'] = total_genii_conversations
                
                with col2:
                    # Multi-select pour les événements de conversion finale
                    active_data = processor.get_active_data()
                    available_events = sorted(active_data['name'].unique()) if not active_data.empty else []
                    
                    if available_events:
                        genii_conversion_events = st.multiselect(
                            "Événement(s) de conversion finale",
                            options=available_events,
                            default=[],
                            key="genii_conversion_events",
                            help="Sélectionnez un ou plusieurs événements qui marquent une conversion réussie"
                        )
                    else:
                        genii_conversion_events = []
                        st.warning("Aucun événement disponible dans les données")
                
                # Séparateur après configuration
                st.markdown("---")
                
                # === NOUVELLE SECTION : PERFORMANCE DE CONVERSION ===
                if genii_conversion_events:
                    display_conversion_performance_section(
                        processor,
                        genii_conversion_events,
                        total_genii_conversations if total_genii_conversations > 0 else None
                    )
                    
                    # Séparateur avant métriques globales
                    st.markdown("---")
                
                # Affichage des métriques globales (avec données filtrées et paramètres Genii)
                display_global_metrics(
                    processor, 
                    total_genii_conversations if total_genii_conversations > 0 else None,
                    genii_conversion_events if genii_conversion_events else None
                )
                
                # Affichage de l'analyse de conversion (nouveau niveau intermédiaire)
                display_conversion_analysis(processor)
                
                # Affichage de l'explorateur d'événements
                display_event_explorer(processor)
            else:
                st.warning("⚠️ Veuillez sélectionner une période valide pour continuer l'analyse")
            
        else:
            st.error(message)


if __name__ == "__main__":
    main()
