import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import font_manager
import os
import urllib.request
import re
import numpy as np

def generar_waterfall_chart_interactivo(categorias, valores, totales, titulo=None):
    """
    Genera un gráfico Waterfall interactivo a partir de listas de categorías, valores y totales.

    Parámetros:
    - categorias: Lista de nombres de las categorías.
    - valores: Lista de valores asociados a cada categoría.
    - totales: Lista booleana indicando si cada categoría es un total.
               Si es total, el valor debe ser un 0.

    Retorno:
    - No retorna valores. Muestra el gráfico generado utilizando plotly.
    """
    # Validación de que las listas tienen la misma longitud
    if not (len(categorias) == len(valores) == len(totales)):
        raise ValueError("Las listas 'categorias', 'valores' y 'totales' deben tener la misma longitud.")

    # Guardar los valores originales para las etiquetas
    valores_originales = valores.copy()

    # Identificación del Total Final: Suma directa de los valores
    suma_total = 0
    for i in range(len(valores)):
        if not totales[i]:
            suma_total += valores[i]
        else:
            valores[i] = suma_total  # Asignar el acumulativo a las filas de totales

    # Definir colores para las barras según las especificaciones
    colores = []
    for i in range(len(valores)):
        if totales[i]:
            colores.append('#004481')  # Azul oscuro para valores iniciales, totales y subtotales
        elif valores[i] >= 0:
            colores.append('#1973b8')  # Azul claro para incrementos
        else:
            colores.append('#d8be75')  # Dorado para decrementos

    # Crear el gráfico Waterfall usando plotly
    fig = go.Figure(go.Waterfall(
        x=categorias,
        y=valores,
        measure=['relative' if not totales[i] else 'total' for i in range(len(valores))],
        text=[f'{int(val)}' for val in valores],  # Usar los valores acumulativos para las etiquetas
        textposition='outside',
        connector={"line": {"color": "white"}},  # No mostrar líneas entre barras
        decreasing={"marker": {"color": '#d8be75'}},  # Dorado para decrementos
        increasing={"marker": {"color": '#1973b8'}},  # Azul claro para incrementos
        totals={"marker": {"color": '#004481'}}  # Azul oscuro para totales
    ))

    # Personalizar el layout del gráfico
    fig.update_layout(
        title=titulo if titulo else "Gráfico Waterfall",  # Usar el título proporcionado si existe
        waterfallgap=0.3,
        xaxis_title="Categoría",
        yaxis_title="Valor",
        plot_bgcolor='white',  # Fondo blanco
        paper_bgcolor='white',  # Fondo del gráfico en blanco
        showlegend=False
    )

    # Guardar el gráfico como un archivo HTML interactivo
    #fig.write_html("waterfall_chart_interactivo.html")

    # Mostrar el gráfico
    #fig.show()
    return fig

def generar_waterfall_chart_interactivo_desde_csv(categories, values, totals):
    """
    Genera un gráfico Waterfall interactivo a partir de un archivo CSV.

    Parámetros:
    - ruta_csv: Ruta al archivo CSV que contiene las categorías, valores y totales.

    Retorno:
    - No retorna valores. Muestra el gráfico generado utilizando plotly.
    """
    # Leer el archivo CSV y renombrar las columnas
    #df = pd.read_csv(ruta_csv)
    #df.columns = ['Categoría', 'Valor', 'Total']  # Renombrar las columnas

    # Convertir la columna 'Total' a booleano
    #df['Total'] = df['Total'].astype(bool)

    #categories = df['Categoría'].tolist()
    #values = df['Valor'].tolist()
    #totals = df['Total'].tolist()

    # Validación de que 'categories', 'values' y 'totals' tienen la misma longitud
    if not (len(categories) == len(values) == len(totals)):
        raise ValueError("Las columnas en el archivo CSV deben tener la misma longitud.")

    # Guardar los valores originales para las etiquetas
    values_original = values.copy()

    # Identificación del Total Final: Suma directa de los valores
    total_sum = 0
    for i in range(len(values)):
        if not totals[i]:
            total_sum += values[i]
        else:
            values[i] = total_sum  # Asignar el acumulativo a las filas de totales

    # Definir colores para las barras según las especificaciones
    colors = []
    for i in range(len(values)):
        if totals[i]:
            colors.append('#004481')  # Azul oscuro para valores iniciales, totales y subtotales
        elif values[i] >= 0:
            colors.append('#1973b8')  # Azul claro para incrementos
        else:
            colors.append('#d8be75')  # Dorado para decrementos

    # Crear el gráfico Waterfall usando plotly
    fig = go.Figure(go.Waterfall(
        x=categories,
        y=values,
        measure=['relative' if not totals[i] else 'total' for i in range(len(values))],
        text=[f'{int(val)}' for val in values],  # Usar los valores acumulativos para las etiquetas
        textposition='outside',
        connector={"line": {"color": "white"}},  # No mostrar líneas entre barras
        decreasing={"marker": {"color": '#d8be75'}},  # Dorado para decrementos
        increasing={"marker": {"color": '#1973b8'}},  # Azul claro para incrementos
        totals={"marker": {"color": '#004481'}}  # Azul oscuro para totales
    ))

    # Personalizar el layout del gráfico
    fig.update_layout(
        title="Gráfico Waterfall",
        waterfallgap=0.3,
        xaxis_title="Categoría",
        yaxis_title="Valor",
        plot_bgcolor='white',  # Fondo blanco
        paper_bgcolor='white',  # Fondo del gráfico en blanco
        showlegend=False
    )

    # Guardar el gráfico como archivo PNG
    # Guardar el gráfico como un archivo HTML interactivo
    fig.write_html("waterfall_chart_interactivo.html")

    # Mostrar el gráfico
    #fig.show()
    return fig

# Función para graficar los gastos sobre ingresos
def plot_ingresos_gastos(ingresos=[], gastos=[], fechas=[], titulo=None):
    # Verificar que ingresos y gastos tengan la misma longitud
    if len(ingresos) != len(gastos):
        raise ValueError("Las listas de ingresos y gastos deben tener la misma longitud.")

    # Cálculo de gastos/ingresos
    ratios = [gasto / ingreso for ingreso, gasto in zip(ingresos, gastos)]

    # Ver si coincide número de fechas con número de datos de ingresos
    if len(fechas) != len(ingresos):
        raise ValueError("La lista de fechas debe tener la misma longitud que las listas de ingresos y gastos.")

    # Crear la figura y los ejes
    #fig, ax = plt.subplots(facecolor='white')

    # Descargar la fuente 'Lato' y agregarla a matplotlib
    font_url = 'https://github.com/google/fonts/raw/main/ofl/lato/Lato-Regular.ttf'
    font_path = 'Lato-Regular.ttf'

    if not os.path.exists(font_path):
        urllib.request.urlretrieve(font_url, font_path)

    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Lato'

    # Crear la figura y los ejes
    fig, ax = plt.subplots(facecolor='white', figsize=(10, 5))

    # Ajustar los márgenes para dejar más espacio entre la gráfica y el eje x
    plt.subplots_adjust(bottom=0.2)  # Incrementa el margen inferior al 20% de la figura

    # Convertir fechas a números para plotting
    x_values = np.arange(len(fechas))

    # Identificar índices de fechas que incluyen "Presupuesto" o "Budget"
    presupuesto_indices = [i for i, fecha in enumerate(fechas) if re.search(r'Presupuesto|Budget', fecha, re.IGNORECASE)]

    # Datos para la línea (excluyendo "Presupuesto" o "Budget")
    non_presupuesto_indices = [i for i in range(len(fechas)) if i not in presupuesto_indices]
    x_line = x_values[non_presupuesto_indices]
    y_line = [ratios[i] for i in non_presupuesto_indices]

    # Plotear la línea con el color especificado
    ax.plot(x_line, y_line, marker='o', linestyle='-', color='#5BBEFF')

    # Añadir etiquetas de datos en cada punto de la línea
    # Definir un umbral para decidir la posición del label
    umbral = 0.5  # Puedes ajustar este valor según tus datos

    for x, y in zip(x_line, y_line):
        # Colocar el label por encima si y es mayor que el umbral, de lo contrario por debajo
        if y > umbral:
            ax.text(x, y - 0.1, f'{y:.2f}', ha='center', color='#072146', fontweight='bold', fontsize=12)
        else:
            ax.text(x, y + 0.1, f'{y:.2f}', ha='center', color='#072146', fontweight='bold', fontsize=12)

    # Si hay fechas con "Presupuesto" o "Budget", añadir los puntos sueltos
    if presupuesto_indices:
        # Usar la posición x del último punto de la línea
        last_x = x_line[-1] if len(x_line) > 0 else x_values[0]
        for idx in presupuesto_indices:
            y_presupuesto = ratios[idx]
            # Añadir el punto en la misma posición x que el último dato
            ax.plot(last_x, y_presupuesto, marker='o', color='#5BBEFF')
            # Añadir etiqueta con el valor del ratio
            ax.text(last_x, y_presupuesto - 0.07, f'{y_presupuesto:.2f}', ha='center', va='bottom', color='#072146', fontweight='bold', fontsize=12)
            # Añadir el texto de la fecha encima del punto
            ax.text(last_x, y_presupuesto - 0.15, fechas[idx], ha='center', va='top', color='#072146', fontsize=12)

    # Configurar el eje x con las etiquetas de fechas (excluyendo "Presupuesto" o "Budget")
    fechas_x = [fechas[i] for i in non_presupuesto_indices]
    ax.set_xticks(x_line)
    ax.set_xticklabels(fechas_x, color='#072146', fontsize=16)
    #ax.set_xticks(range(len(fechas)))  # Set x-ticks to match the number of data points
    #ax.set_xticklabels(fechas, rotation=45, ha='right')

    # Ajustar la posición de las etiquetas del eje x hacia abajo
    ax.tick_params(axis='x', pad=30)

    # Invertir el eje y
    ax.invert_yaxis()

    # Eliminar el eje y
    ax.get_yaxis().set_visible(False)

    # Establecer el fondo blanco
    ax.set_facecolor('white')

    # Eliminar el recuadro (marco) de la gráfica
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Eliminar la línea y las marcas del eje x
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', which='both', length=0)

    # Establecer el título si se proporciona, alineado a la izquierda y con el color especificado
    if titulo:
        plt.title(titulo, loc='left', color='#072146', fontsize=16)

    # Guardar la imagen **antes** de mostrarla
    #plt.savefig('output_image.png', bbox_inches='tight')

    # Mostrar la gráfica
    #plt.show()

    # Cerrar la figura para liberar memoria
    #plt.close(fig)
    return fig

def app():
    st.title("Chart Generator")

    # Menú para elegir el tipo de gráfico
    graph_type = st.selectbox("Seleccione el tipo de gráfico", ["Waterfall Chart", "Ingresos vs Gastos"])

    if graph_type == "Waterfall Chart":
        # Inputs para categorías, valores y totales
        categories_input = st.text_input("Categorías (separadas por comas):")
        values_input = st.text_input("Valores (separados por comas):")
        totals_input = st.text_input("Totales (separados por comas, True/False):")

        # Botón para generar el gráfico
        generate_button = st.button("Generar Gráfico")

        if generate_button:
            # Procesar inputs
            categories = [category.strip() for category in categories_input.split(",")]
            values = [float(value) for value in values_input.split(",")]
            totals = [value.lower() == "true" for value in totals_input.split(",")]

            # Llamar a la función generar_waterfall_chart
            fig = generar_waterfall_chart_interactivo(categories, values, totals)

            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig, height=600)

            # Botón para descargar la imagen
            img_data = BytesIO()
            fig.write_image(img_data, format='png')  # Especificar el formato de la imagen
            img_data.seek(0)

            st.download_button(
                label="Descargar imagen",
                data=img_data,
                file_name="waterfall_chart.png",
                mime="image/png"
            )

            # Botón para descargar en formato HTML
            html_data = BytesIO()
            html_data.write(fig.to_html(full_html=False).encode())
            html_data.seek(0)

            st.download_button(
                label="Descargar en HTML",
                data=html_data,
                file_name="waterfall_chart.html",
                mime="text/html"
            )

    elif graph_type == "Ingresos vs Gastos":
        # Inputs para ingresos, gastos y fechas
        titulo_input = st.text_input("Título del gráfico (opcional):")
        ingresos_input = st.text_input("Ingresos (separados por comas):")
        gastos_input = st.text_input("Gastos (separados por comas):")
        fechas_input = st.text_input("Fechas (separadas por comas):")

        # Botón para generar el gráfico
        generate_button = st.button("Generar Gráfico")

        if generate_button:
            # Procesar inputs
            ingresos = [float(ingreso) for ingreso in ingresos_input.split(",")]
            gastos = [float(gasto) for gasto in gastos_input.split(",")]
            fechas = [fecha.strip() for fecha in fechas_input.split(",")]

            # Llamar a la función plot_ingresos_gastos
            fig = plot_ingresos_gastos(ingresos, gastos, fechas, titulo_input)

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)

            # Botón para descargar la imagen
            img_data = BytesIO()
            plt.savefig(img_data, format='png', bbox_inches='tight')
            img_data.seek(0)

            st.download_button(
                label="Descargar imagen",
                data=img_data,
                file_name="ingresos_gastos.png",
                mime="image/png"
            )

if __name__ == "__main__":
    app()
