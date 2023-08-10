import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from abc_analysis import abc_analysis, abc_plot
from dash.dependencies import Input, Output
import dash_auth
from babel.numbers import format_currency
from dash import dcc
from dash import html
from dash import dash_table
import locale

USERNAME_PASSWORD_PAIRS = [['username','password'], ['agirsaude','gsup']]

# import from folders/theme changer
from app import *
from dash_bootstrap_templates import ThemeSwitchAIO

#app.py
#auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)
import dash

FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]

app = dash.Dash(__name__, external_stylesheets=FONT_AWESOME)
server = app.server
app.scripts.config.serve_locally = True



# ========== Styles ============ #
tab_card = {'height': '100%'}

main_config = {
    "hovermode": "x unified",
    "legend": {"yanchor":"top", 
                "y":0.9, 
                "xanchor":"left",
                "x":0.1,
                "title": {"text": None},
                "font" :{"color":"white"},
                "bgcolor": "rgba(0,0,0,0.5)"},
    "margin": {"l":10, "r":10, "t":10, "b":10}
}

config_graph={"displayModeBar": False, "showTips": False}

template_theme1 = "flatly"
template_theme2 = "darkly"
url_theme1 = dbc.themes.FLATLY
url_theme2 = dbc.themes.DARKLY


# ===== Reading n cleaning File ====== #
df = pd.read_csv("assets/trimestre1.csv")
df_cru = df.copy()

# Transformando em inteiros e  retirando o cifrão R$
df['mes'] = df['mes'].astype(int)
df['ano'] = df['ano'].astype(int)
df['trimestre'] = df['trimestre'].astype(int)

df['valor_total'] = df['valor_total'].str.replace('R\$\s*', '', regex=True)  # Remove currency symbol
df['valor_total'] = df['valor_total'].str.replace('\.', '', regex=True)     # Remove thousands separators
df['valor_total'] = df['valor_total'].str.replace(',', '.', regex=True)     # Replace commas with dots
df['valor_total'] = df['valor_total'].astype(float)
df['valor_unitario'] = df['valor_unitario'].str.replace('R\$\s*', '', regex=True)  # Remove currency symbol
df['valor_unitario'] = df['valor_unitario'].str.replace('\.', '', regex=True)     # Remove thousands separators
df['valor_unitario'] = df['valor_unitario'].str.replace(',', '.', regex=True)     # Replace commas with dots
df['valor_unitario'] = df['valor_unitario'].astype(float)
df['cp_cs'] = df['cp_cs'].replace({'OCS': 'SERVIÇOS', 'OCP': 'PRODUTOS'})

sum_df1 = df.groupby('especie')['valor_total'].sum().reset_index(name='sum_especie_total')
sum_df2 = df.groupby('classe')['valor_total'].sum().reset_index(name='sum_classe_total')
sum_df3 = df.groupby('subclasse')['valor_total'].sum().reset_index(name='sum_subclasse_total')
sum_df4 = df.groupby('item')['valor_total'].sum().reset_index(name='sum_item_total')



df1 = sum_df1.sort_values(by=['sum_especie_total'], ascending=False)
df2 = sum_df2.sort_values(by=['sum_classe_total'], ascending=False)
df3 = sum_df3.sort_values(by=['sum_subclasse_total'], ascending=False)
df4 = sum_df4.sort_values(by=['sum_item_total'], ascending=False)


# Criando opções pros filtros que virão
options_trimestre = [{'label': 'Todos trimestres', 'value': 0}]
for i, j in zip(df_cru['trimestre'].unique(), df['trimestre'].unique()):
    options_trimestre.append({'label': i, 'value': j})
options_trimestre = sorted(options_trimestre, key=lambda x: x['value']) 

options_entidade = [{'label': 'Todas Entidades', 'value': 0}]
for i in df['entidade'].unique():
    options_entidade.append({'label': i, 'value': i})
# ========= Função dos Filtros ========= #
def trimestre_filter(trimestre):
    if trimestre == 0:
        mask = df['trimestre'].isin(df['trimestre'].unique())
    else:
        mask = df['trimestre'].isin([trimestre])
    return mask

def entidade_filter(entidade):
    if entidade == 0:
        mask = df['entidade'].isin(df['entidade'].unique())
    else:
        mask = df['entidade'].isin([entidade])
    return mask

def convert_to_text(trimestre):
    if trimestre == 0:
        x = 'Todos trimestres'
    elif trimestre == 1:
        x = 'Primeiro trimestre de 2023'
    elif trimestre == 2:
        x = 'Segundo trimestre de 2023'
    elif trimestre == 3:
        x = 'Terceiro trimestre de 2023'
    elif trimestre == 4:
        x = '3'
    else:
        x = 'Trimestre inválido'
     
    return x



# =========  Layout  =========== #
app.layout = dbc.Container(children=[
    # Armazenamento de dataset
    # dcc.Store(id='dataset', data=df_store),

    # Layout
    # Row 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([  
                            html.Legend("Análise de Compras")
                        ], sm=8),
                        dbc.Col([        
                            html.I(className='fa fa-balance-scale', style={'font-size': '300%'})
                        ], sm=4, align="center")
                    ]),
                    dbc.Row([
                        dbc.Col([
                            ThemeSwitchAIO(aio_id="theme", themes=[url_theme1, url_theme2]),
                            html.Legend("AGIR")
                        ])
                    ], style={'margin-top': '10px'}),
                    dbc.Row([
                        dbc.Button("Fonte dos dados", href="https://docs.google.com/spreadsheets/d/1tvJSPQku9YOdKdP2HMs6BtjqfDGfhsvRW5kPDMyCNMk/edit#gid=118474863", target="_blank")
                    ], style={'margin-top': '10px'})
                ])
            ], style=tab_card)
        ], sm=4, lg=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row(
                        dbc.Col(
                            html.Legend('Compras por trimestre: (0 = último de 2022)')
                        )
                    ),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph1', className='dbc', config=config_graph)
                        ], sm=12, md=7),
                        dbc.Col([
                            dcc.Graph(id='graph2', className='dbc', config=config_graph)
                        ], sm=12, lg=5)
                    ])
                ])
            ], style=tab_card)
        ], sm=12, lg=7),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row(
                        dbc.Col([
                            html.H5('Escolha o trimestre'),
                            dbc.RadioItems(
                                id="radio-trimestre",
                                options=options_trimestre,
                                value=0,
                                inline=True,
                                labelCheckedClassName="text-success",
                                inputCheckedClassName="border border-success bg-success",
                            ),
                            html.Div(id='trimestre-select', style={'text-align': 'center', 'margin-top': '30px'}, className='dbc')
                        ])
                    )
                ])
            ], style=tab_card)
        ], sm=12, lg=3)
    ], className='g-2 my-auto', style={'margin-top': '7px'}),

    # Row 2
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='graph3', className='dbc', config=config_graph)
                        ])
                    ], style=tab_card)
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='graph4', className='dbc', config=config_graph)
                        ])
                    ], style=tab_card)
                ])
            ], className='g-2 my-auto', style={'margin-top': '7px'})
        ], sm=12, lg=5),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='graph5', className='dbc', config=config_graph)
                        ])
                    ], style=tab_card)
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='graph7', className='dbc', config=config_graph)
                        ])
                    ], style=tab_card)
                ])
            ], className='g-2 my-auto', style={'margin-top': '7px'})
        ], sm=12, lg=5),
        dbc.Col([
            dbc.Card([
                dcc.Graph(id='graph8', className='dbc', config=config_graph)
            ], style=tab_card)
        ], sm=20, lg=2)
    ], className='g-2 my-auto', style={'margin-top': '7px'}),
    
    # Row 3
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4('6 items de maior gasto'),
                    dcc.Graph(id='graph9', className='dbc', config=config_graph)
                ])
            ], style=tab_card)
        ], sm=10, lg=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("6 subclasses de maior gasto"),
                    dcc.Graph(id='graph10', className='dbc', config=config_graph)
                ])
            ], style=tab_card)
        ], sm=10, lg=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='graph11', className='dbc', config=config_graph)
                ])
            ], style=tab_card)
        ], sm=10, lg=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Escolha a Entidade'),
                    dbc.RadioItems(
                        id="radio-entidade",
                        options=options_entidade,
                        value=0,
                        inline=True,
                        labelCheckedClassName="text-warning",
                        inputCheckedClassName="border border-warning bg-warning",
                    ),
                    html.Div(id='entidade-select', style={'text-align': 'center', 'margin-top': '30px'}, className='dbc')
                ])
            ], style=tab_card)
        ], sm=10, lg=2),
    ], className='g-2 my-auto', style={'margin-top': '7px'}),
    
        # New Row (Row 4) for duplicated graphs
    dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(id='graph12', className='dbc', config=config_graph),
                dcc.Graph(id='graph13', className='dbc', config=config_graph)
            ])
        ], style=tab_card)
    ], sm=10, lg=5),
    dbc.Col([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id='graph14', className='dbc', config=config_graph)
                    ])
                ], style=tab_card)
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id='graph15', className='dbc', config=config_graph)
                    ])
                ], style={**tab_card, 'height': '100%'}),  # Adjust the height of graph15
            ])
        ], className='g-2 my-auto', style={'margin-top': '7px'})
    ], sm=12, lg=4),
    dbc.Col([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id='graph16', className='dbc', config=config_graph)
                    ])
                ], style={**tab_card, 'height': '200%'}),  # Adjust the height of graph16 to be twice as tall
            ])
        ]),
    ], sm=24, lg=3)
], className='g-2 my-auto', style={'margin-top': '7px'})  # end of New Row (Row 4)












], fluid=True, style={'height': '100vh'})




# ======== Callbacks ========== #
# Graph 1 and 2
import plotly.graph_objects as go

@app.callback(
    Output('graph1', 'figure'),
    Output('graph2', 'figure'),
    Output('trimestre-select', 'children'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph1(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_1 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_1 = df_1.loc[mask]

    # Calculate the total sum of 'valor_total'
    total_valor = df_1['valor_total'].sum()

    df2 = df_1.groupby(['valor_total']).sum().reset_index()
    df_1 = df_1.groupby(['trimestre'])['valor_total'].sum().reset_index()

    df_1['formatted_valor_total'] = 'R$ ' + df_1['valor_total'].astype(str)

    fig2 = go.Figure()
    fig2.add_trace(go.Indicator(mode='number',
        title={"text": f"<span style='font-size:150%'>Valor Total</span><br><span style='font-size:70%'>Em Reais</span><br>"},
        value=total_valor,  # Use the calculated total_valor here
        number={'prefix': "R$"}
    ))

    fig1 = go.Figure(go.Bar(x=df_1['trimestre'], y=df_1['valor_total'], textposition='auto', text=df_1['formatted_valor_total']))
    fig2.update_layout(main_config, height=300, template=template)
    fig1.update_layout(main_config, height=200, template=template)

    select = html.H1("Todas Entidades") if entidade == 0 else html.H1(entidade)

    return fig1, fig2, select





def update_xaxes(fig, top_n=10):
    # Show only the top N groups on the X-axis initially
    fig.update_xaxes(range=[0, top_n - 1])
    return fig

# Graph 3
@app.callback(
    Output('graph3', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph9(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_3 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_3 = df_3.loc[mask]

    df_3 = df_3.groupby('especie')['valor_total'].sum().reset_index(name='sum_especie_total')

    # Sort the dataframe by 'sum_especie_total' in descending order
    df_3 = df_3.sort_values(by='sum_especie_total', ascending=False)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=df_3['especie'], y=df_3['sum_especie_total'], name='Valor Total'))
    fig3.add_trace(go.Scatter(x=df_3['especie'], y=df_3['sum_especie_total'].cumsum(), mode='lines', name='Cumulative Total'))
    fig3.update_layout(
        main_config,
        height=270,
        template=template,
        xaxis_title='ESPECIE',
        yaxis_title='Valor Total',
        xaxis_tickangle=-45,  # Adjust the angle of X-axis labels
        margin=dict(l=50, r=10, t=10, b=50)  # Adjust the margins around the graph area
    )
    fig3 = update_xaxes(fig3, top_n=10)  # Show only the top 10 groups initially
    return fig3

# Graph 4
@app.callback(
    Output('graph4', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph9(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_4 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_4 = df_4.loc[mask]

    df_4 = df_4.groupby('classe')['valor_total'].sum().reset_index(name='sum_classe_total')

    # Sort the dataframe by 'sum_classe_total' in descending order
    df_4 = df_4.sort_values(by='sum_classe_total', ascending=False)

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=df_4['classe'], y=df_4['sum_classe_total'], name='Valor Total'))
    fig4.add_trace(go.Scatter(x=df_4['classe'], y=df_4['sum_classe_total'].cumsum(), mode='lines', name='Cumulative Total'))
    fig4.update_layout(
        main_config,
        height=270,
        template=template,
        xaxis_title='CLASSE',
        yaxis_title='Valor Total',
        xaxis_tickangle=-45,  # Adjust the angle of X-axis labels
        margin=dict(l=50, r=10, t=10, b=50)  # Adjust the margins around the graph area
    )
    fig4 = update_xaxes(fig4, top_n=10)  # Show only the top 10 groups initially
    return fig4

# Graph 5 
@app.callback(
    Output('graph5', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph9(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_5 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_5 = df_5.loc[mask]

    df_5 = df_5.groupby('subclasse')['valor_total'].sum().reset_index(name='sum_subclasse_total')

    # Sort the dataframe by 'sum_subclasse_total' in descending order
    df_5 = df_5.sort_values(by='sum_subclasse_total', ascending=False)

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(x=df_5['subclasse'], y=df_5['sum_subclasse_total'], name='Valor Total'))
    fig5.add_trace(go.Scatter(x=df_5['subclasse'], y=df_5['sum_subclasse_total'].cumsum(), mode='lines', name='Cumulative Total'))
    fig5.update_layout(
        main_config,
        height=270,
        template=template,
        xaxis_title='SUBCLASSE',
        yaxis_title='Valor Total',
        xaxis_tickangle=-45,  # Adjust the angle of X-axis labels
        margin=dict(l=50, r=10, t=10, b=50)  # Adjust the margins around the graph area
    )
    fig5 = update_xaxes(fig5, top_n=10)  # Show only the top 10 groups initially
    return fig5

# Graph 7
@app.callback(
    Output('graph7', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph9(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_7 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_7 = df_7.loc[mask]

    df_7 = df_7.groupby('item')['valor_total'].sum().reset_index(name='sum_item_total')

    # Sort the dataframe by 'sum_item_total' in descending order
    df_7 = df_7.sort_values(by='sum_item_total', ascending=False)

    fig7 = go.Figure()
    fig7.add_trace(go.Bar(x=df_7['item'], y=df_7['sum_item_total'], name='Valor Total'))
    fig7.add_trace(go.Scatter(x=df_7['item'], y=df_7['sum_item_total'].cumsum(), mode='lines', name='Cumulative Total'))
    fig7.update_layout(
        main_config,
        height=270,
        template=template,
        xaxis_title='ITEM',
        yaxis_title='Valor Total',
        xaxis_tickangle=-45,  # Adjust the angle of X-axis labels
        margin=dict(l=50, r=10, t=10, b=50)  # Adjust the margins around the graph area
    )
    fig7 = update_xaxes(fig7, top_n=10)  # Show only the top 10 groups initially
    return fig7

@app.callback(
    Output('graph8', 'figure'),
    Input('radio-trimestre', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph8(trimestre, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_8 = df.loc[mask]

    df_8 = df_8.groupby('entidade')['valor_total'].sum().reset_index()
    fig8 = go.Figure(go.Bar(
        x=df_8['entidade'],  # Swap x and y axes
        y=df_8['valor_total'],  # Swap x and y axes
        orientation='v',  # Set orientation to 'v' for vertical bars
        textposition='auto',
        text=df_8['valor_total'],
        insidetextfont=dict(family='Times', size=15)))

    fig8.update_layout(main_config, height=360, template=template)
    return fig8


# Graph 9
@app.callback(
    Output('graph9', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph9(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_9 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_9 = df_9.loc[mask]

    df_9 = df_9.groupby('item')['valor_total'].sum().reset_index()

    # Get the top 5 items
    df9_top5 = df_9.sort_values(by='valor_total', ascending=False).head(6)
    top5_items = df9_top5['item'].tolist()

    # Filter the original DataFrame for the top 5 items
    df_9 = df[df['item'].isin(top5_items)]

    fig9 = px.bar(df_9, y="valor_total", x="trimestre", color="item")  # Using Plotly Express

    fig9.update_layout(main_config, height=250, template=template)

    # Update the legend position and display mode
    fig9.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=-1.9999, xanchor='left', x=0),
        showlegend=True,
    )

    return fig9




# Graph 10
@app.callback(
    Output('graph10', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph10(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_10 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_10 = df_10.loc[mask]

    df_10 = df_10.groupby('subclasse')['valor_total'].sum().reset_index()

    # Get the top 6 items
    df10_top5 = df_10.sort_values(by='valor_total', ascending=False).head(6)
    top5_items = df10_top5['subclasse'].tolist()

    # Filter the original DataFrame for the top 6 items
    df_10 = df[df['subclasse'].isin(top5_items)]

    fig10 = px.bar(df_10, y="valor_total", x="trimestre", color="subclasse")  # Using Plotly Express

    fig10.update_layout(main_config, height=250, template=template)

    # Update the legend position and display mode
    fig10.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=-1.82, xanchor='left', x=0),
        showlegend=True,
    )

    return fig10

# Graph 11
@app.callback(
    Output('graph11', 'figure'),
    Output('entidade-select', 'children'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph11(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_11 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_11 = df_11.loc[mask]

    df11 = df_11.groupby(['valor_total']).sum().reset_index()
    fig11 = go.Figure()
    fig11.add_trace(go.Indicator(mode='number',
        title = {"text": f"<span style='font-size:150%'>Valor Total</span><br><span style='font-size:70%'>Em Reais</span><br>"},
        value = df_11['valor_total'].sum(),  # Corrected line: use df_11 instead of df
        number = {'prefix': "R$"}
    ))

    fig11.update_layout(main_config, height=300, template=template)
    select = html.H1("Todas Entidades") if entidade == 0 else html.H1(entidade)

    return fig11, select


@app.callback(
    Output('graph13', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph13(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_13 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_13 = df_13.loc[mask]

    # Filter the DataFrame to include only rows where 'cp_cs' is 'SERVIÇOS'
    df_13 = df_13[df_13['cp_cs'] == 'SERVIÇOS']

    df_13 = df_13.groupby('item')['valor_total'].sum().reset_index(name='sum_item_total')

    # Sort the dataframe by 'sum_item_total' in descending order
    df_13 = df_13.sort_values(by='sum_item_total', ascending=False)

    fig13 = go.Figure()
    fig13.add_trace(go.Bar(x=df_13['item'], y=df_13['sum_item_total'], name='Valor Total'))
    fig13.add_trace(go.Scatter(x=df_13['item'], y=df_13['sum_item_total'].cumsum(), mode='lines', name='Cumulative Total'))
    fig13.update_layout(
        main_config,
        height=360,
        template=template,
        xaxis_title='ITEM - SERVICOS',
        yaxis_title='Valor Total',
        xaxis_tickangle=-45,  # Adjust the angle of X-axis labels
        margin=dict(l=50, r=10, t=40, b=20)  # Adjust the margins around the graph area
    )
    fig13 = update_xaxes(fig13, top_n=10)  # Show only the top 10 groups initially
    return fig13

@app.callback(
    Output('graph12', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph12(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_12 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_12 = df_12.loc[mask]

    dfpie = df_12.groupby('cp_cs')['valor_total'].sum().reset_index()

    fig12 = go.Figure()
    fig12.add_trace(go.Pie(labels=dfpie['cp_cs'], values=dfpie['valor_total'], hole=0.6))
    fig12.update_layout(
        main_config,
        template=template,
        margin=dict(l=10, r=10, t=10, b=10)  # Adjust the margins around the pie chart
    )
    return fig12

@app.callback(
    Output('graph14', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph14(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_14 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_14 = df_14.loc[mask]

    item_counts = df_14.groupby('item')['numero_oc'].nunique().reset_index()
    item_counts.columns = ['item', 'item_count']
    sorted_df = item_counts.sort_values(by='item_count', ascending=False)
    top_10 = sorted_df.head(10)

    fig14 = px.bar(top_10, x='item', y='item_count', title='',
                  labels={'item': 'Item', 'item_count': 'N de ordem de compra'})

    fig14.update_xaxes(range=[top_10['item'].min(), top_10['item'].max()])
    fig14.update_layout(
        main_config,
        height=450,
        template=template,
        xaxis_tickangle=-45,  # Adjust the angle of X-axis labels
        margin=dict(l=50, r=10, t=30, b=30)  # Adjust the margins around the graph area
    )
    return fig14

# Graph 15
import plotly.express as px

@app.callback(
    Output('graph15', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph15(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_15 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_15 = df_15.loc[mask]

    # Step 1: Calculate the amplitude for each item and the result in %
    item_counts = df_15['item'].value_counts()
    popular_items = item_counts[item_counts > 10].index
    filtered_df = df_15[df_15['item'].isin(popular_items)]
    amplitudes = filtered_df.groupby('item')['valor_unitario'].apply(lambda x: (x.max() - x.min()) / x.mean() * 100)

    # Step 2: Create an interactive vertical bar chart with the top 10 highest %
    top_amplitudes = amplitudes.nlargest(10).reset_index()

    fig15 = px.bar(top_amplitudes, x='item', y='valor_unitario', title='')
    fig15.update_xaxes(categoryorder='total descending')
    fig15.update_layout(
        main_config,
        template=template,
        xaxis_title='Item',
        yaxis_title='Amplitude dos preços em relação à média%',
        xaxis_tickangle=-45,  # Adjust the angle of X-axis labels
        margin=dict(l=50, r=10, t=10, b=50)  # Adjust the margins around the graph area
    )
    return fig15

@app.callback(
    Output('graph16', 'figure'),
    Input('radio-trimestre', 'value'),
    Input('radio-entidade', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph16(trimestre, entidade, toggle):
    template = template_theme1 if toggle else template_theme2

    mask = trimestre_filter(trimestre)
    df_16 = df.loc[mask]

    mask = entidade_filter(entidade)
    df_16 = df_16.loc[mask]

    # Calculate the sum of "valor_total" for each item and get the top 50 highest sums
    top_items = df_16.groupby('item')['valor_total'].sum().nlargest(50).index

    # Create a new DataFrame for the final table
    new_df = pd.DataFrame({'item': top_items})

    # Calculate "quantidade de oc"
    quantidade_de_oc = df_16.groupby('item')['numero_oc'].nunique()
    new_df['quantidade de oc'] = new_df['item'].map(quantidade_de_oc)

    # Calculate "valor mínimo"
    valor_minimo = df_16.groupby('item')['valor_unitario'].min()
    new_df['valor minimo'] = new_df['item'].map(valor_minimo)

    # Calculate "valor médio"
    valor_medio = df_16.groupby('item')['valor_unitario'].mean()
    new_df['valor médio'] = new_df['item'].map(valor_medio)

    # Calculate "valor máximo"
    valor_maximo = df_16.groupby('item')['valor_unitario'].max()
    new_df['valor máximo'] = new_df['item'].map(valor_maximo)

    # Calculate "desvio padrão"
    desvio_padrao = df_16.groupby('item')['valor_unitario'].std()
    new_df['desvio padrão'] = new_df['item'].map(desvio_padrao)

    new_df_sorted = new_df.sort_values(by='quantidade de oc', ascending=False)

   
    top_20 = new_df_sorted.head(20)

    # Set Brazilian currency format for the specified columns
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
    currency_columns = ['valor minimo', 'valor médio', 'valor máximo', 'desvio padrão']
    for col in currency_columns:
        top_20[col] = top_20[col].apply(lambda x: locale.currency(x, grouping=True))

    fig16 = go.Figure(data=[go.Table(
        header=dict(values=['25 Itens com > valor de compras e > qde de OC', 'Quantidade de OC', 'Valor Mínimo', 'Valor Médio', 'Valor Máximo', 'Desvio Padrão']),
        cells=dict(values=[top_20['item'], top_20['quantidade de oc'], top_20['valor minimo'], top_20['valor médio'], top_20['valor máximo'], top_20['desvio padrão']])
    )])
    fig16.update_layout(
        main_config,
        template=template,
        margin=dict(l=10, r=10, t=10, b=10)  # Adjust the margins around the table
    )
    return fig16

# Run server
if __name__ == '__main__':
    app.run_server(debug=False)


