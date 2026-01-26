import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import warnings
warnings.filterwarnings('ignore')

def procesar_archivo(archivo, hoja_nombre, max_clientes, umbral_balance, max_iter):
    """Procesa el archivo Excel y genera grupos de clientes optimizados"""
    try:
        df = pd.read_excel(archivo.name, sheet_name=hoja_nombre)
        
        required_cols = ["Cliente", "Producto", "Cantidad"]
        if not all(col in df.columns for col in required_cols):
            return None, f"❌ Error: El archivo debe contener las columnas: {required_cols}", None
        
        if not pd.api.types.is_numeric_dtype(df["Cantidad"]):
            return None, "❌ Error: La columna 'Cantidad' debe contener valores numéricos", None
        
        tabla = df.pivot_table(index="Cliente", columns="Producto", values="Cantidad", fill_value=0)
        clientes = tabla.index.tolist()
        n_clientes = len(clientes)
        
        scaler = StandardScaler()
        tabla_scaled = scaler.fit_transform(tabla)
        tabla_scaled_df = pd.DataFrame(tabla_scaled, index=clientes)
        
        n_groups = max(2, (n_clientes + max_clientes - 1) // max_clientes)
        
        def calcular_afinidad_promedio(tabla, clientes_grupo):
            if len(clientes_grupo) < 2:
                return 0.0
            subset = tabla.loc[clientes_grupo]
            similitudes = cosine_similarity(subset)
            n = similitudes.shape[0]
            suma_similitud = np.sum(similitudes) - n
            cantidad_pares = n * (n - 1)
            return (suma_similitud / cantidad_pares) * 100 if cantidad_pares > 0 else 0.0
        
        def calcular_afinidad_ponderada(tabla, df, clientes_grupo):
            if len(clientes_grupo) < 2:
                return 0.0
            subset = tabla.loc[clientes_grupo]
            cantidades = df[df["Cliente"].isin(clientes_grupo)].groupby("Cliente")["Cantidad"].sum().reindex(clientes_grupo).values
            similitudes = cosine_similarity(subset)
            n = similitudes.shape[0]
            suma_ponderada = 0.0
            peso_total = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    peso = cantidades[i] * cantidades[j]
                    suma_ponderada += similitudes[i, j] * peso
                    peso_total += peso
            return (suma_ponderada / peso_total) * 100 if peso_total > 0 else 0.0
        
        def calcular_productos_unicos(tabla, clientes_grupo):
            subset = tabla.loc[clientes_grupo]
            return (subset > 0).any().sum()
        
        def calcular_suma_cantidades(df, clientes_grupo):
            return df[df["Cliente"].isin(clientes_grupo)]["Cantidad"].sum()
        
        def evaluar_configuracion(tabla, df, grupos_dict):
            score = 0
            n_groups = len(set(grupos_dict.values()))
            suma_por_grupo = {}
            for grupo in range(n_groups):
                clientes_grupo = [c for c, g in grupos_dict.items() if g == grupo]
                if not clientes_grupo:
                    continue
                afinidad = calcular_afinidad_promedio(tabla, clientes_grupo)
                productos_unicos = calcular_productos_unicos(tabla, clientes_grupo)
                suma_cantidades = calcular_suma_cantidades(df, clientes_grupo)
                suma_por_grupo[grupo] = suma_cantidades
                score += afinidad - 50 * productos_unicos
            
            if suma_por_grupo:
                max_suma = max(suma_por_grupo.values())
                min_suma = min(suma_por_grupo.values())
                balance_penalty = 1000 * (max_suma - min_suma) / (max_suma + 1e-6)
                score -= balance_penalty
            return score / n_groups if n_groups > 0 else 0
        
        np.random.seed(42)
        kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        grupos = kmeans.fit_predict(tabla_scaled)
        cliente_to_grupo = {cliente: grupo for cliente, grupo in zip(clientes, grupos)}
        
        for grupo in range(n_groups):
            clientes_grupo = [c for c, g in cliente_to_grupo.items() if g == grupo]
            if len(clientes_grupo) > max_clientes:
                exceso = clientes_grupo[max_clientes:]
                for cliente in exceso:
                    for g in range(n_groups):
                        if g != grupo and sum(1 for c, gr in cliente_to_grupo.items() if gr == g) < max_clientes:
                            cliente_to_grupo[cliente] = g
                            break
        
        suma_total = df["Cantidad"].sum()
        objetivo_suma = suma_total / n_groups
        mejor_configuracion = cliente_to_grupo.copy()
        mejor_score = evaluar_configuracion(tabla, df, cliente_to_grupo)
        
        progreso = f"🤖 Optimizando grupos...\n"
        progreso += f"📊 Clientes: {n_clientes} | Grupos: {n_groups} | Score inicial: {mejor_score:.2f}\n\n"
        
        mejoras = 0
        for i in range(max_iter):
            nuevo_grupo = cliente_to_grupo.copy()
            clientes_a_mover = random.sample(clientes, min(10, n_clientes))
            
            for cliente in clientes_a_mover:
                current_grupo = nuevo_grupo[cliente]
                posibles_grupos = [g for g in range(n_groups) if sum(1 for c, gr in nuevo_grupo.items() if gr == g) < max_clientes]
                if posibles_grupos and len(posibles_grupos) > 1:
                    nuevo_grupo[cliente] = random.choice([g for g in posibles_grupos if g != current_grupo])
            
            suma_por_grupo = {}
            for grupo in range(n_groups):
                clientes_grupo = [c for c, g in nuevo_grupo.items() if g == grupo]
                suma_por_grupo[grupo] = calcular_suma_cantidades(df, clientes_grupo)
            
            max_suma = max(suma_por_grupo.values()) if suma_por_grupo else 0
            min_suma = min(suma_por_grupo.values()) if suma_por_grupo else 0
            
            if max_suma == 0 or abs(max_suma - min_suma) <= objetivo_suma * umbral_balance:
                score = evaluar_configuracion(tabla, df, nuevo_grupo)
                if score > mejor_score:
                    mejor_score = score
                    mejor_configuracion = nuevo_grupo.copy()
                    mejoras += 1
            else:
                max_grupo = max(suma_por_grupo, key=suma_por_grupo.get)
                min_grupo = min(suma_por_grupo, key=suma_por_grupo.get)
                clientes_grande = [c for c, g in nuevo_grupo.items() if g == max_grupo]
                cantidades_clientes = df.groupby("Cliente")["Cantidad"].sum().loc[clientes_grande].to_dict()
                centroide_chico = tabla_scaled_df.loc[[c for c, g in nuevo_grupo.items() if g == min_grupo]].mean()
                distancias = np.linalg.norm(tabla_scaled_df.loc[clientes_grande].values - centroide_chico.values, axis=1)
                clientes_ordenados = [(c, cantidades_clientes[c], distancias[i]) for i, c in enumerate(clientes_grande)]
                clientes_ordenados.sort(key=lambda x: (x[1], x[2]))
                suma_a_mover = (suma_por_grupo[max_grupo] - suma_por_grupo[min_grupo]) / 2
                suma_movida = 0
                clientes_a_mover = []
                for cliente, cantidad, _ in clientes_ordenados:
                    if len([c for c, g in nuevo_grupo.items() if g == min_grupo]) >= max_clientes:
                        break
                    if suma_movida + cantidad <= suma_a_mover:
                        clientes_a_mover.append(cliente)
                        suma_movida += cantidad
                    if suma_movida >= suma_a_mover * 0.9:
                        break
                for cliente in clientes_a_mover:
                    nuevo_grupo[cliente] = min_grupo
                score = evaluar_configuracion(tabla, df, nuevo_grupo)
                if score > mejor_score:
                    mejor_score = score
                    mejor_configuracion = nuevo_grupo.copy()
                    mejoras += 1
        
        progreso += f"✅ Optimización completa: {mejoras} mejoras encontradas\n"
        progreso += f"🎯 Score final: {mejor_score:.2f}\n"
        
        tabla["Grupo"] = [mejor_configuracion[cliente] for cliente in tabla.index]
        tabla_clientes = df.pivot_table(index="Cliente", columns="Producto", values="Cantidad", fill_value=0)
        tabla_clientes["Grupo"] = tabla["Grupo"]
        
        resumen_data = []
        for grupo in range(n_groups):
            clientes_grupo = tabla_clientes[tabla_clientes["Grupo"] == grupo].index
            if not clientes_grupo.empty:
                afinidad = calcular_afinidad_promedio(tabla_clientes.drop("Grupo", axis=1), clientes_grupo)
                afinidad_ponderada = calcular_afinidad_ponderada(tabla_clientes.drop("Grupo", axis=1), df, clientes_grupo)
                productos_unicos = calcular_productos_unicos(tabla_clientes.drop("Grupo", axis=1), clientes_grupo)
                suma_cantidades = calcular_suma_cantidades(df, clientes_grupo)
                resumen_data.append({
                    "Grupo": f"Grupo {grupo}",
                    "Afinidad %": round(afinidad, 2),
                    "Afinidad % Ponderada": round(afinidad_ponderada, 2),
                    "Cantidad Clientes": len(clientes_grupo),
                    "Productos Únicos": productos_unicos,
                    "Suma Cantidades": round(suma_cantidades, 2)
                })
        
        resumen_afinidad = pd.DataFrame(resumen_data)
        archivo_salida = "GruposResultado_Optimizado.xlsx"
        
        with pd.ExcelWriter(archivo_salida, engine="xlsxwriter") as writer:
            for grupo in range(n_groups):
                clientes_grupo = tabla[tabla["Grupo"] == grupo].index
                if not clientes_grupo.empty:
                    clientes_grupo_df = pd.DataFrame({"Cliente": clientes_grupo})
                    clientes_grupo_df.to_excel(writer, sheet_name=f"Grupo_{grupo}", index=False)
            resumen_afinidad.to_excel(writer, sheet_name="Resumen", index=False)
        
        progreso += f"\n📦 {n_groups} grupos creados exitosamente"
        return archivo_salida, progreso, resumen_afinidad
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}", None

with gr.Blocks(theme=gr.themes.Soft(), title="Sistema de Agrupación de Clientes") as demo:
    gr.Markdown("""
    # 📊 Sistema de Agrupación de Clientes por Afinidad
    
    Agrupa automáticamente clientes según sus patrones de compra usando Machine Learning.
    
    ### 📋 Requisitos del archivo:
    - Formato: **Excel (.xlsx)**
    - Columnas obligatorias: `Cliente`, `Producto`, `Cantidad`
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            archivo_input = gr.File(label="📤 Sube tu archivo Excel", file_types=[".xlsx", ".xls"])
            with gr.Accordion("⚙️ Configuración Avanzada", open=False):
                hoja_input = gr.Textbox(value="Hoja1", label="Nombre de la hoja")
                max_clientes_input = gr.Slider(10, 200, 70, 5, label="Máximo de clientes por grupo")
                umbral_input = gr.Slider(0.05, 0.30, 0.10, 0.05, label="Umbral de balance")
                iteraciones_input = gr.Slider(50, 300, 100, 10, label="Iteraciones de optimización")
            procesar_btn = gr.Button("🚀 Procesar Archivo", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            progreso_output = gr.Textbox(label="📊 Progreso", lines=10)
            resumen_output = gr.Dataframe(label="📈 Resumen de Grupos")
            archivo_output = gr.File(label="📥 Descargar Resultado")
    
    gr.Markdown("""
    ---
    ### 💡 Interpretación de Resultados:
    - **Afinidad %**: Similaridad promedio entre clientes (mayor es mejor)
    - **Afinidad % Ponderada**: Afinidad considerando volumen de compras
    - **Cantidad Clientes**: Número de clientes en cada grupo
    - **Productos Únicos**: Diversidad de productos (menor es mejor)
    - **Suma Cantidades**: Total de unidades compradas por el grupo
    """)
    
    procesar_btn.click(
        fn=procesar_archivo,
        inputs=[archivo_input, hoja_input, max_clientes_input, umbral_input, iteraciones_input],
        outputs=[archivo_output, progreso_output, resumen_output]
    )

if __name__ == "__main__":
    demo.launch()