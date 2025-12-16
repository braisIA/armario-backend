import pandas as pd
import os

def create_full_dataset():
    """
    Une los archivos users.csv, garments.csv, contexts.csv y outfit_ratings.csv
    para generar el dataset completo de entrenamiento.
    """
    print("\n--- Preparando el Dataset de Entrenamiento ---")

    # Archivos esperados
    csv_files = ['users.csv', 'garments.csv', 'contexts.csv', 'outfit_ratings.csv']
    for file in csv_files:
        if not os.path.exists(file):
            print(f"❌ ERROR: No se encuentra '{file}'. Asegúrate de ejecutar primero el generador de datos.")
            return None

    # --- 1. Cargar los CSV ---
    print("1️⃣  Cargando archivos CSV...")
    users = pd.read_csv('users.csv')
    garments = pd.read_csv('garments.csv')
    contexts = pd.read_csv('contexts.csv')
    ratings = pd.read_csv('outfit_ratings.csv')

    # --- 2. Estandarizar nombres de columnas clave ---
    if 'id' in users.columns and 'user_id' not in users.columns:
        users.rename(columns={'id': 'user_id'}, inplace=True)

    if 'id' in contexts.columns and 'context_id' not in contexts.columns:
        contexts.rename(columns={'id': 'context_id'}, inplace=True)

    if 'id' in garments.columns:
        garments.rename(columns={'id': 'garment_id'}, inplace=True)

    # --- 3. Unir tablas ---
    print("2️⃣  Uniendo tablas para crear el dataset completo...")

    # Join con users
    df_full = pd.merge(ratings, users, on='user_id', how='left')

    # Join con contexts
    df_full = pd.merge(df_full, contexts, on='context_id', how='left')

    # Join con garments (top)
    df_full = pd.merge(
        df_full,
        garments.add_prefix('top_'),
        left_on='top_garment_id',
        right_on='top_garment_id',
        how='left'
    )

    # Join con garments (bottom)
    df_full = pd.merge(
        df_full,
        garments.add_prefix('bottom_'),
        left_on='bottom_garment_id',
        right_on='bottom_garment_id',
        how='left'
    )

    # Join con garments (shoes)
    df_full = pd.merge(
        df_full,
        garments.add_prefix('shoes_'),
        left_on='shoes_garment_id',
        right_on='shoes_garment_id',
        how='left'
    )

    # --- 4. Limpieza y orden ---
    print("3️⃣  Limpiando y reorganizando columnas...")

    # Eliminar columnas duplicadas o confusas si aparecen
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]

    # --- 5. Guardar dataset final ---
    print("4️⃣  Guardando dataset final...")

    output_file = 'full_training_dataset.csv'
    df_full.to_csv(output_file, index=False)

    print(f"\n✅ Dataset completo creado con éxito: '{output_file}'")
    print(f"➡️  Filas: {df_full.shape[0]}, Columnas: {df_full.shape[1]}")
    print("\nColumnas principales:", list(df_full.columns)[:15], "...")
    return df_full


if __name__ == '__main__':
    create_full_dataset()
