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

# Classes pour la gestion des données
class ROIDataProcessor:
    """Classe pour traiter et analyser les données ROI"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.processed_data = None
        self.consolidated_data = None
        self.events_analysis = {}
        
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
                    
                    # Parse du contenu si au format clé:valeur, sinon utiliser la colonne comme clé
                    if ':' in str(value_content):
                        parts = str(value_content).split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                        else:
                            key = key_from_column
                            value = str(value_content).strip()
                    else:
                        key = key_from_column
                        value = str(value_content).strip()
                    
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
    
    def get_global_metrics(self) -> Dict:
        """Calcule les métriques globales"""
        if self.processed_data is None:
            self.processed_data = self.parse_values_column()
            
        # Conversion de la colonne date
        self.processed_data['date'] = pd.to_datetime(self.processed_data['date'], errors='coerce')
        
        total_events = len(self.processed_data)
        unique_events = self.processed_data['name'].nunique()
        date_range = (
            self.processed_data['date'].min(),
            self.processed_data['date'].max()
        )
        
        # Répartition par type d'événement
        event_distribution = self.processed_data['name'].value_counts()
        
        # Répartition par type de valeur
        value_type_distribution = self.processed_data['value_key'].value_counts()
        
        return {
            'total_events': total_events,
            'unique_events': unique_events,
            'date_range': date_range,
            'event_distribution': event_distribution,
            'value_type_distribution': value_type_distribution
        }
    
    def analyze_selected_events(self, selected_events: List[str]) -> Dict:
        """Analyse les événements sélectionnés"""
        if self.processed_data is None:
            self.processed_data = self.parse_values_column()
            
        # Filtrage des données
        filtered_data = self.processed_data[
            self.processed_data['value_key'].isin(selected_events)
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
        
        # Évolution temporelle du CA
        price_data_sorted = price_data.sort_values('date')
        price_data_sorted['cumulative_revenue'] = price_data_sorted['value_numeric'].cumsum()
        
        return {
            'type': 'price',
            'total_revenue': total_revenue,
            'avg_transaction': avg_transaction,
            'transaction_count': transaction_count,
            'temporal_data': price_data_sorted,
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


def display_global_metrics(processor: ROIDataProcessor):
    """Affiche les métriques globales (Niveau 1)"""
    st.markdown('<div class="section-header">📊 Vue Généraliste - Métriques Globales</div>', 
                unsafe_allow_html=True)
    
    metrics = processor.get_global_metrics()
    
    # Métriques principales
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
        daily_events = processor.processed_data.groupby(
            processor.processed_data['date'].dt.date
        ).size().reset_index()
        daily_events.columns = ['date', 'count']
        
        fig_timeline = px.line(
            daily_events, 
            x='date', 
            y='count',
            title="Nombre d'événements par jour",
            labels={'count': 'Nombre d\'événements', 'date': 'Date'}
        )
        st.plotly_chart(fig_timeline, use_container_width=True)


def display_event_explorer(processor: ROIDataProcessor):
    """Affiche l'explorateur d'événements (Niveau 2)"""
    st.markdown('<div class="section-header">🔍 Explorateur d\'Événements</div>', 
                unsafe_allow_html=True)
    
    if processor.processed_data is None:
        processor.processed_data = processor.parse_values_column()
    
    # Liste des noms d'événements disponibles (pas les value_key, mais les 'name')
    available_event_names = sorted(processor.processed_data['name'].unique())
    
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
            # Comptage du nombre d'occurrences pour info
            event_count = len(processor.processed_data[processor.processed_data['name'] == event_name])
            
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
    
    # S'assurer que les données sont parsées
    if processor.processed_data is None:
        processor.processed_data = processor.parse_values_column()
    
    # Filtrage des données par nom d'événement
    filtered_data = processor.processed_data[
        processor.processed_data['name'].isin(selected_event_names)
    ].copy()
    
    if filtered_data.empty:
        st.warning("Aucune donnée trouvée pour les événements sélectionnés")
        return
    
    # Conversion des dates si pas déjà fait
    filtered_data['date'] = pd.to_datetime(filtered_data['date'], errors='coerce')
    
    # === MÉTRIQUES PRINCIPALES ===
    total_events = len(filtered_data)
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
    
    # Comptage par jour et par nom d'événement
    daily_events = filtered_data.groupby([
        filtered_data['date'].dt.date, 'name'
    ]).size().unstack(fill_value=0).reset_index()
    
    # Si un seul type d'événement, graphique simple
    if len(selected_event_names) == 1:
        daily_simple = filtered_data.groupby(
            filtered_data['date'].dt.date
        ).size().reset_index()
        daily_simple.columns = ['date', 'count']
        
        fig_timeline = px.line(
            daily_simple,
            x='date',
            y='count',
            title=f"Évolution - {selected_event_names[0]}",
            labels={'count': 'Nombre d\'événements', 'date': 'Date'}
        )
    else:
        # Plusieurs événements : graphique empilé
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
                value=f"{total_revenue:.2f}€",
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
                value=f"{avg_transaction:.2f}€",
                delta=None
            )
        
        # Évolution du CA cumulé
        st.subheader("📈 Évolution du Chiffre d'Affaires Cumulé")
        
        price_data_sorted = price_data.sort_values('date')
        price_data_sorted['cumulative_revenue'] = price_data_sorted['value_numeric'].cumsum()
        
        fig_revenue = px.line(
            price_data_sorted,
            x='date',
            y='cumulative_revenue',
            title="Chiffre d'Affaires Cumulé",
            labels={'cumulative_revenue': 'CA Cumulé (€)', 'date': 'Date'}
        )
        
        # Ajout de markers pour les points de vente
        fig_revenue.add_scatter(
            x=price_data_sorted['date'],
            y=price_data_sorted['cumulative_revenue'],
            mode='markers',
            name='Transactions',
            marker=dict(size=8, color='red', symbol='circle')
        )
        
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Répartition des prix si plusieurs valeurs
        if transaction_count > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💵 Répartition des Prix")
                price_distribution = price_data['value_numeric'].value_counts().sort_index()
                
                fig_price_dist = px.bar(
                    x=price_distribution.index,
                    y=price_distribution.values,
                    title="Distribution des Prix",
                    labels={'x': 'Prix (€)', 'y': 'Nombre de Transactions'}
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
        event_counts = filtered_data['name'].value_counts()
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
            row_data['CA Associé (€)'] = f"{revenue:.2f}"
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
            st.metric("CA Total", f"{total_revenue_all:.2f}€")
        with col4:
            avg_revenue = total_revenue_all / total_count if total_count > 0 else 0
            st.metric("CA Moyen", f"{avg_revenue:.2f}€")
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
            st.write(f"• **{best_value}**: {best_revenue:.2f}€ ({best_count} occurrences)")
        
        with ranking_col2:
            if len(revenue_ranking) > 1:
                st.write("**📈 Panier moyen par valeur:**")
                for value, revenue in revenue_ranking:
                    count = value_counts[value]
                    avg_basket = revenue / count if count > 0 else 0
                    st.write(f"• **{value}**: {avg_basket:.2f}€/transaction")
    
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
                value=f"{total_revenue:.2f}€",
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
                value=f"{avg_transaction:.2f}€",
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
        if 'distribution' in analysis:
            fig_price_dist = px.bar(
                x=analysis['distribution'].index,
                y=analysis['distribution'].values,
                title=f"Distribution des prix - {event}",
                labels={'x': 'Prix (€)', 'y': 'Fréquence'}
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
            
            # Affichage des métriques globales
            display_global_metrics(processor)
            
            # Affichage de l'explorateur d'événements
            display_event_explorer(processor)
            
        else:
            st.error(message)


if __name__ == "__main__":
    main()
