import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate
from dash import dcc, html
from dash import exceptions
import pandas as pd
import numpy as np
import base64
import io
import os

class TEGAnalyzer:
    def __init__(self, df):
        self.teg_model_name = 'TEG Model'  # Placeholder for model name
        self.df = df
        self.dwell_dfs = {}
        self.process_data()
        self.average_peak = self.compute_average_peak()
        
    def process_data(self):
        self.df['Time'] = pd.to_datetime(self.df['Time'])
        self.df['TEG Power'] = self.df['TEG_Voltage'] * self.df['TEG_Current']
        self.df['T_Delta'] = self.df['T_hot'] - self.df['T_cold']
        self.df['T_hot'] = np.round(self.df['T_hot'])
        self.df['T_cold'] = np.round(self.df['T_cold'])
        self.df['Elapsed Time'] = self.df.groupby('Cycle ID')['Time'].diff().dt.total_seconds().fillna(0).groupby(self.df['Cycle ID']).cumsum() / 60
        self.df['Resistance'] = self.df['TEG_Voltage'] / self.df['TEG_Current']
        self.cycle_dfs = {cycle: self.df[self.df['Cycle ID'] == cycle] for cycle in self.df['Cycle ID'].unique()}

    def compute_average_peak(self):
        summary_data = []
        ref_cycle_id = 2
        ref_data = None

        for cycle, cycle_df in self.cycle_dfs.items():
            filtered_df = cycle_df[(cycle_df['Elapsed Time'] >= 6) & (cycle_df['Elapsed Time'] <= 13)]
            avg_voltage = filtered_df['TEG_Voltage'].mean()
            avg_current = filtered_df['TEG_Current'].mean()
            avg_power = filtered_df['TEG Power'].mean()
            avg_resistance = filtered_df['Resistance'].mean()
            max_power = cycle_df['TEG Power'].max()
            cycle_duration = cycle_df['Elapsed Time'].iloc[-1]

            summary = [cycle, avg_voltage, avg_current, avg_power, avg_resistance, max_power, cycle_duration]

            # Storing reference cycle data for later use
            if cycle == ref_cycle_id:
                ref_data = summary

            summary_data.append(summary)

        average_peak_df = pd.DataFrame(summary_data, columns=['Cycle ID', 'Average TEG Voltage', 'Average TEG Current', 'Average TEG Power', 'Average Resistance', 'Maximum Power', 'Cycle Duration'])

        # Calculate percentage difference using the reference data
        if ref_data:
            ref_power = ref_data[3]  # Average TEG Power for reference cycle
            ref_resistance = ref_data[4]  # Average Resistance for reference cycle

            average_peak_df['TEG Power % Difference w.r.t Cycle 2'] = ((average_peak_df['Average TEG Power'] - ref_power) / ref_power) * 100
            average_peak_df['Resistance % Difference w.r.t Cycle 2'] = ((average_peak_df['Average Resistance'] - ref_resistance) / ref_resistance) * 100
        
        return average_peak_df
    
    
class TEG_System:
    def __init__(self, df):
        self.df = df
        self.TEG_Model = 'TEG System Model'  # Placeholder for model name
        self.preprocessed_data = self.preprocess_data()
        self.system_fidelity_df = self.compute_phases()
        
    def preprocess_data(self):
        df = self.df.copy()
        df['T_hot'] = df['T_hot'].round(0)
        df['T_cold'] = df['T_cold'].round(0)
        df['Time'] = pd.to_datetime(df['Time'])
        df['TEG_Power'] = df['TEG_Voltage'] * df['TEG_Current']
        df['T_Delta'] = df['T_hot'] - df['T_cold']
        df['Elapsed Time'] = df.groupby('Cycle ID')['Time'].transform(lambda x: (x - x.iloc[0]).dt.total_seconds() / 60)
        return df

    def compute_phases(self):
        results = []
        for cycle in self.preprocessed_data['Cycle ID'].unique():
            cycle_data = self.preprocessed_data[self.preprocessed_data['Cycle ID'] == cycle]
            
            heating_end_time = cycle_data[cycle_data['T_hot'] == 190]['Time'].min()
            heating_start_time = cycle_data[cycle_data['T_hot'] == 40]['Time'].min()

            heating_duration = (heating_end_time - heating_start_time).seconds / 60 if heating_end_time and heating_start_time else 0

            dwell_end_time = cycle_data[(cycle_data['T_hot'] <= 210) & (cycle_data['T_hot'] >= 190)]['Time'].max()
            dwell_start_time = cycle_data[(cycle_data['T_hot'] <= 210) & (cycle_data['T_hot'] >= 190)]['Time'].min()

            dwell_duration = (dwell_end_time - dwell_start_time).seconds / 60 if dwell_end_time and dwell_start_time else 0

            active_phase = heating_duration + dwell_duration

            cycle_duration = cycle_data['Elapsed Time'].max()

            cooling_duration = cycle_duration - active_phase

            results.append([cycle, heating_duration, dwell_duration, cooling_duration])
        
        return pd.DataFrame(results, columns=['Cycle ID', 'Heating Phase Duration', 'Dwell Phase Duration', 'Cooling Phase Duration'])


def parse_content(contents, filenames):
    dfs = []
    truncated_names = []
    for content, filename in zip(contents, filenames):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
                dfs.append(df)
                truncated_names.append(filename)
        except Exception as e:
            print(e)
            return None
    return dfs, truncated_names

def plot_average_teg_power_dcc(dfs, filenames):
    color_palette = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'brown', 'pink']
    fig = go.Figure()
    for idx, (df, filename) in enumerate(zip(dfs, filenames)):
        analyzer = TEGAnalyzer(df)
        df_avg = analyzer.average_peak
        
        df_avg['Cycle ID'] = pd.to_numeric(df_avg['Cycle ID'], errors='coerce')
        df_avg.dropna(subset=['Cycle ID'], inplace=True)
        
        # Use consistent color from the palette for each file
        file_color = color_palette[idx % len(color_palette)]
        
        fig.add_trace(go.Scatter(x=df_avg['Cycle ID'], y=df_avg['Average TEG Power'], mode='lines+markers', name=filename.split('.')[0], line=dict(color=file_color)))
        
        # Add horizontal line at 1W
        fig.add_shape(type="line", x0=df_avg['Cycle ID'].min(), x1=df_avg['Cycle ID'].max(), y0=1, y1=1, line=dict(color="green", width=2, dash="dash"))
        
    # Styling
    fig.update_layout(
        plot_bgcolor="white", 
        paper_bgcolor="white", 
        title="Average TEG Power for Accelerated TEG Thermal Cycling",
        xaxis_title="Cycle ID",
        yaxis_title="Power (W)"
    )
    
    fig.update_layout(height=700, width=1500)
    return fig

def performance_curve_dcc(df, filename, selected_cycle_id):
    analyzer = TEGAnalyzer(df)
    reference_cycle_2 = analyzer.df[analyzer.df['Cycle ID'] == 2]
    cycle_data = analyzer.df[analyzer.df['Cycle ID'] == selected_cycle_id]
    
    fig = make_subplots(rows=3, cols=1, subplot_titles=("TEG Power vs T_Delta", "TEG Voltage vs T_Delta", "TEG Current vs T_Delta"))
    
    fig.add_trace(go.Scatter(x=cycle_data['T_Delta'], y=cycle_data['TEG Power'], mode='markers+lines', line=dict(color='blue'), name=f'TEG Performance for Cycle ID {selected_cycle_id}'), row=1, col=1)
    fig.add_trace(go.Scatter(x=reference_cycle_2['T_Delta'], y=reference_cycle_2['TEG Power'], mode='markers', marker=dict(color='green'), name='Reference Cycle 2'), row=1, col=1)
    fig.add_shape(go.layout.Shape(type='line', y0=1, y1=1, x0=min(cycle_data['T_Delta']), x1=max(cycle_data['T_Delta']), line=dict(color="red", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', dash="dash"), name='1W Mark'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=cycle_data['T_Delta'], y=cycle_data['TEG_Voltage'], mode='markers+lines', line=dict(color='blue'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=reference_cycle_2['T_Delta'], y=reference_cycle_2['TEG_Voltage'], mode='markers', marker=dict(color='green'), showlegend=False), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=cycle_data['T_Delta'], y=cycle_data['TEG_Current'], mode='markers+lines', line=dict(color='blue'), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=reference_cycle_2['T_Delta'], y=reference_cycle_2['TEG_Current'], mode='markers', marker=dict(color='green'), showlegend=False), row=3, col=1)
    
    fig.update_layout(
        plot_bgcolor="white", 
        paper_bgcolor="white",
        width=1500,
        height=1200,
        title=f"TEG Performance - Cycle ID {selected_cycle_id} for {filename.split('.')[0]}"
    )
    return fig

def plot_system_fidelity_dcc(dfs, filenames):
    graphs = []
    for df, filename in zip(dfs, filenames):
        system = TEG_System(df)
        system_fidelity_df = system.system_fidelity_df.copy()
        system_fidelity_df['Cycle ID'] = pd.to_numeric(system_fidelity_df['Cycle ID'], errors='coerce')
        
        fig = go.Figure()
        for col in ['Heating Phase Duration', 'Dwell Phase Duration', 'Cooling Phase Duration']:
            fig.add_trace(go.Scatter(x=system_fidelity_df['Cycle ID'], 
                                     y=system_fidelity_df[col], 
                                     mode='lines+markers', 
                                     name=col))
        
        fig.update_layout(
            title=f"{filename.split('.')[0]}; System Fidelity",
            xaxis_title="Cycle ID",
            yaxis_title="Phase Duration in minutes",
            legend_title="Phases",
            autosize=False,
            width=1500,
            height=800,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        graphs.append(dcc.Graph(figure=fig))
    return fig

global_files_data = {}

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("TEG Thermal Cycling Analyzer", style={"textAlign": "center"}),
    html.Hr(),
    html.H2("TEG Performance", style={"textAlign": "center"}),
    html.Hr(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Excel Files')
        ]),
        multiple=True
    ),
    html.Ul(id='output-filenames'),
    html.Button('Compute and Plot', id='compute-btn'),
    html.Hr(),
    dcc.Graph(id='average-teg-power-plot'),
    dcc.Graph(id='performance-curve-plot'),
    html.Hr(),
    html.H2("System Fidelity: Heating, Dwell and Cooling Phase Durations", style={"textAlign": "center"}),
    html.Hr(),
    dcc.Graph(id='system-fidelity-plot')
])

def parse_contents(contents, filenames):
    dfs = []
    for content in contents:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded))
        dfs.append(df)
    return dfs, filenames

@app.callback(
    [Output('output-filenames', 'children'),
     Output('average-teg-power-plot', 'figure')],
    [Input('compute-btn', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename')]
)
def update_output(n_clicks, contents, filenames):
    if contents is None:
        raise exceptions.PreventUpdate

    dfs, filenames = parse_contents(contents, filenames)
    global_files_data['dfs'] = dfs
    global_files_data['filenames'] = filenames

    # Creating a list of uploaded filenames
    children = [html.Li(filename) for filename in filenames]

    fig = plot_average_teg_power_dcc(dfs, filenames)
    return children, fig

@app.callback(
    [Output('performance-curve-plot', 'figure'),
     Output('system-fidelity-plot', 'figure')],
    [Input('average-teg-power-plot', 'clickData')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename')]
)
def on_point_click(click_data, contents, filenames):
    if not click_data:
        raise exceptions.PreventUpdate

    selected_cycle_id = click_data['points'][0]['x']
    selected_file_idx = click_data['points'][0]['curveNumber']

    df = global_files_data['dfs'][selected_file_idx]
    filename = global_files_data['filenames'][selected_file_idx]

    performance_curve_fig = performance_curve_dcc(df, filename, selected_cycle_id)

    # Fetch the system fidelity for the clicked file
    system_fidelity_fig = plot_system_fidelity_dcc([df], [filename])

    return performance_curve_fig, system_fidelity_fig

if __name__ == '__main__':
    app.run_server(debug=True, port = 8065)
