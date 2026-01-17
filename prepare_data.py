"""
Script pour télécharger et préparer les VRAIES données ISO Open Data
URLs officielles depuis https://www.iso.org/open-data.html
"""

import pandas as pd
import requests
from pathlib import Path
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import sys
import json
import ast

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# VRAIES URLs ISO OPEN DATA (officielles)
# =============================================================================

ISO_OPEN_DATA_URLS = {
    'standards': {
        'csv': 'https://isopublicstorageprod.blob.core.windows.net/opendata/_latest/iso_deliverables_metadata/csv/iso_deliverables_metadata.csv',
    },
    'committees': {
        'csv': 'https://isopublicstorageprod.blob.core.windows.net/opendata/_latest/iso_technical_committees/csv/iso_technical_committees.csv',
    },
    'ics': {
        'csv': 'https://isopublicstorageprod.blob.core.windows.net/opendata/_latest/iso_ics/csv/ICS.csv'
    }
}

# =============================================================================
# 1. TÉLÉCHARGEMENT DES DONNÉES ISO
# =============================================================================

def download_iso_data():
    """Télécharge les datasets ISO Open Data (format CSV)"""
    
    Path('data').mkdir(exist_ok=True)
    
    # Standards et committees sont essentiels, ICS est optionnel
    essential_datasets = {
        'standards': ISO_OPEN_DATA_URLS['standards']['csv'],
        'committees': ISO_OPEN_DATA_URLS['committees']['csv']
    }
    
    optional_datasets = {
        'ics': ISO_OPEN_DATA_URLS['ics']['csv']
    }
    
    success = True
    
    # Téléchargement des datasets essentiels
    for name, url in essential_datasets.items():
        try:
            logger.info(f"Downloading {name} from ISO Open Data...")
            
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            csv_path = f'data/{name}.csv'
            with open(csv_path, 'wb') as f:
                f.write(response.content)
            
            # Lecture avec gestion d'erreurs
            df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
            
            parquet_path = f'data/{name}.parquet'
            df.to_parquet(parquet_path)
            
            logger.info(f"✓ {name}: {len(df)} rows downloaded and converted to Parquet")
            
        except Exception as e:
            logger.error(f"✗ Failed to download {name}: {e}")
            success = False
    
    # Téléchargement des datasets optionnels (non bloquant)
    for name, url in optional_datasets.items():
        try:
            logger.info(f"Downloading {name} (optional)...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            csv_path = f'data/{name}.csv'
            with open(csv_path, 'wb') as f:
                f.write(response.content)
            
            # ICS a des problèmes de format, on essaie avec error_bad_lines
            df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
            df.to_parquet(f'data/{name}.parquet')
            
            logger.info(f"✓ {name}: {len(df)} rows (optional)")
            
        except Exception as e:
            logger.warning(f"⚠ Could not download {name} (optional): {e}")
            # Pas bloquant, on continue
    
    return success

# =============================================================================
# 2. PRÉPARATION DES DONNÉES STANDARDS
# =============================================================================

def extract_iso_map_regex(text, lang='en'):
    """
    Extract value from ISO map format: {key: "value", key2: "value"}
    This handles keys without quotes which JSON/AST cannot parse.
    """
    if pd.isna(text):
        return ''
    text = str(text)
    
    # Pattern to find lang key followed by quoted string
    # Try with specific lang key first
    # Matches: en: "Content here"
    pattern = f'{lang}:\\s*"((?:[^"\\\\]|\\\\.)*)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1).replace('\\"', '"')
    
    # Fallback: simple finding if keys are quoted or logic differs
    return ''

def prepare_standards_data(n=30000):
    """
    Prépare les données des standards ISO pour le RAG
    """
    
    logger.info("Loading ISO deliverables metadata...")
    df = pd.read_parquet('data/standards.parquet')
    
    logger.info(f"Total deliverables: {len(df)}")
    
    # DEBUG: Print column names and sample stage
    logger.info(f"Columns: {df.columns.tolist()}")
    if 'currentStage' in df.columns:
        logger.info(f"Sample stages: {df['currentStage'].unique()[:5]}")
    
    # Filtre: Standards publiés uniquement (stage 60.60)
    # Handle string or int or float
    # Cast to numeric, handle errors
    df['stage_numeric'] = pd.to_numeric(df['currentStage'], errors='coerce')
    
    # Check 60.60 (float) or 6060 (int)
    # The user says "6060" int.
    # We'll check for both representations just in case
    df_pub = df[
        (df['stage_numeric'] == 6060) | 
        (df['stage_numeric'] == 60.60) |
        (df['currentStage'].astype(str).str.contains('60.60'))
    ].copy()
    
    logger.info(f"Published standards (filtered): {len(df_pub)}")
    
    if len(df_pub) == 0:
        logger.warning("Filtering by stage returned 0 rows! Checking top rows of original df:")
        logger.warning(df[['reference', 'currentStage']].head())
        # Fallback to taking everything if filter fails, just for debug/demo
        df_pub = df.copy()
    
    df = df_pub
    
    # Extraction du titre anglais
    logger.info("Extracting titles...")
    
    # Check if we already have flattened columns (dot notation from CSV)
    if 'title.en' in df.columns:
        logger.info("Using pre-existing 'title.en' column.")
        df['title_en'] = df['title.en'].astype(str)
    elif 'title' in df.columns:
        logger.info("Parsing 'title' column with Regex...")
        df['title_en'] = df['title'].apply(lambda x: extract_iso_map_regex(x, 'en'))
    else:
        logger.warning("No 'title' or 'title.en' column found! Using empty strings.")
        df['title_en'] = ''

    # Même chose pour title_fr
    if 'title.fr' in df.columns:
        df['title_fr'] = df['title.fr'].astype(str)
    elif 'title' in df.columns:
        df['title_fr'] = df['title'].apply(lambda x: extract_iso_map_regex(x, 'fr'))
    else:
        df['title_fr'] = ''

    # Même chose pour scope
    if 'scope.en' in df.columns:
        df['scope_text'] = df['scope.en'].astype(str)
    elif 'scope' in df.columns:
        df['scope_text'] = df['scope'].apply(lambda x: extract_iso_map_regex(x, 'en'))
    else:
        df['scope_text'] = ''
    
    # Fallback cleanup
    df['title_en'] = df['title_en'].replace('nan', '').fillna('')
    df['scope_text'] = df['scope_text'].replace('nan', '').fillna('')
    
    # Texte complet pour RAG
    df['full_text'] = (
        df['reference'].fillna('') + ' ' +
        df['title_en'].fillna('') + ' ' +
        df['scope_text'].fillna('')
    )
    
    # Filtre: Garder seulement ceux avec du contenu (au moins 10 chars)
    df = df[df['full_text'].str.len() > 10].copy()
    logger.info(f"Standards with content: {len(df)}")
    
    # Score de pertinence (Optionnel, utile si n < total)
    df['year'] = pd.to_datetime(df['publicationDate'], errors='coerce').dt.year
    df['recency_score'] = (df['year'].fillna(2000) - 1950) / (2026 - 1950)
    
    # Priorité: Standards de management et IT
    priority_ics = ['03.120', '35.', '13.020', '03.100', '27.']
    df['priority_score'] = df['icsCode'].apply(
        lambda x: 1.0 if any(ics in str(x) for ics in priority_ics) else 0.5
    )
    
    # Score final
    df['final_score'] = df['recency_score'] * 0.6 + df['priority_score'] * 0.4
    
    # Sélection top N
    df_selected = df.nlargest(min(n, len(df)), 'final_score')
    
    logger.info(f"\nSelected {len(df_selected)} standards:")
    if not df_selected.empty:
        logger.info(f"- Year range: {df_selected['year'].min():.0f}-{df_selected['year'].max():.0f}")
        logger.info(f"- Unique committees: {df_selected['ownerCommittee'].nunique()}")
        logger.info(f"- Avg text length: {df_selected['full_text'].str.len().mean():.0f} chars")
    
    return df_selected

# =============================================================================
# 3. CRÉATION DE LA BASE SQLite
# =============================================================================

def create_sqlite_db(df_standards):
    """Crée la base SQLite optimisée pour l'application RAG"""
    
    logger.info("Creating SQLite database...")
    
    conn = sqlite3.connect('iso_standards.db')
    
    # On renomme et prépare les colonnes pour l'app
    df_export = df_standards.copy()
    
    # Mapping des colonnes pour matcher l'app
    df_export['id'] = df_export['reference']
    df_export['abstract'] = df_export['scope_text']
    df_export['status'] = 'Published'
    
    # Colonnes à sauvegarder
    cols = [
        'id', 'reference', 'title_en', 'title_fr', 
        'abstract', 'publicationDate', 'edition',
        'icsCode', 'ownerCommittee', 'full_text', 'status', 'year'
    ]
    
    # Intersection avec colonnes existantes
    final_cols = [c for c in cols if c in df_export.columns]
    
    df_export[final_cols].to_sql('standards', conn, if_exists='replace', index=False)
    
    logger.info(f"✓ Standards table: {len(df_export)} rows")
    
    # Table committees (si disponible)
    committees_path = 'data/committees.parquet'
    if Path(committees_path).exists():
        try:
            df_committees = pd.read_parquet(committees_path)
            logger.info(f"Committees columns: {df_committees.columns.tolist()}")
            
            # --- Processing Committees ---
            # 1. Extract Title
            if 'title.en' in df_committees.columns:
                df_committees['title_en'] = df_committees['title.en'].astype(str)
            elif 'title' in df_committees.columns:
                df_committees['title_en'] = df_committees['title'].apply(lambda x: extract_iso_map_regex(x, 'en'))
            else:
                df_committees['title_en'] = ''
            
            # 2. Extract ID (reference -> id)
            if 'reference' in df_committees.columns:
                df_committees['id'] = df_committees['reference']
            
            # Select simple columns for SQL
            # We want: id, title_en, scope (optional)
            comm_cols = ['id', 'reference', 'title_en']
            final_comm_cols = [c for c in comm_cols if c in df_committees.columns]
            
            df_committees[final_comm_cols].to_sql('committees', conn, if_exists='replace', index=False)
            logger.info(f"✓ Committees table: {len(df_committees)} rows")
            
        except Exception as e:
            logger.warning(f"Could not load committees: {e}")
    
    conn.close()
    
    db_size = Path('iso_standards.db').stat().st_size / 1024 / 1024
    logger.info(f"✓ Database created: iso_standards.db ({db_size:.1f} MB)")
    
    return 'iso_standards.db'

# =============================================================================
# 4. GÉNÉRATION DES EMBEDDINGS
# =============================================================================

def prepare_embeddings(df_standards):
    """Génère les embeddings pour la recherche sémantique"""
    
    logger.info("Generating embeddings with SentenceTransformer...")
    logger.info("Loading model 'all-MiniLM-L6-v2'...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = df_standards['full_text'].fillna("").tolist()
    logger.info(f"Encoding {len(texts)} documents (this may take a few minutes)...")
    
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True
    )
    
    # Sauvegarde
    np.save('embeddings.npy', embeddings)
    # Sauvegarde des IDs correspondants (reference sert d'ID)
    df_standards[['reference']].rename(columns={'reference': 'id'}).to_csv('embeddings_ids.csv', index=False)
    
    size_mb = Path('embeddings.npy').stat().st_size / 1024 / 1024
    logger.info(f"✓ Embeddings saved: {embeddings.shape} ({size_mb:.1f} MB)")
    
    return embeddings

# =============================================================================
# 5. SCRIPT PRINCIPAL
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("ISO OPEN DATA - PREPARATION FOR RAG APPLICATION")
    logger.info("Source: https://www.iso.org/open-data.html")
    logger.info("=" * 70)
    
    # Étape 1: Téléchargement
    logger.info("\n[1/4] Downloading ISO Open Data...")
    if not download_iso_data():
        logger.error("Essential downloads failed. Exiting.")
        sys.exit(1)
    
    # Étape 2: Préparation standards
    # n=30000 pour couvrir l'ensemble des standards publiés
    logger.info("\n[2/4] Preparing standards data...")
    df_selected = prepare_standards_data(n=30000)
    
    if df_selected.empty:
        logger.error("No standards to process. Exiting.")
        sys.exit(1)
    
    # Étape 3: Base de données
    logger.info("\n[3/4] Creating SQLite database...")
    create_sqlite_db(df_selected)
    
    # Étape 4: Embeddings
    logger.info("\n[4/4] Generating embeddings...")
    prepare_embeddings(df_selected)
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ PREPARATION COMPLETE")
    logger.info("=" * 70)
    logger.info("\nFiles created:")
    logger.info("- iso_standards.db")
    logger.info("- embeddings.npy")
    logger.info("- embeddings_ids.csv")
    logger.info("\n→ Ready to adapt your RAG application!")
    logger.info("\nAttribution required:")
    logger.info('This work uses iso_deliverables_metadata from ISO Open Data')
    logger.info('Licensed under ODC Attribution License (ODC-By) v1.0')
    logger.info('https://www.iso.org/open-data.html')

if __name__ == "__main__":
    main()
