# Code from: https://plotly.com/python/network-graphs/
import plotly.graph_objects as go
import time

class PlotlyDraw:

    def __init__(self, graph, weights):
        self.G = graph
        self.weights = weights
        self.__build()

    def __build(self):
        edge_x = []
        edge_y = []
        edge_x_center = []
        edge_y_center = []
        edge_text_center = []

        for edge in self.G.edges():
            x0, y0 = self.G.nodes[edge[0]]['pos']
            x1, y1 = self.G.nodes[edge[1]]['pos']
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_x_center.append((x0 + x1) / 2)
            edge_y_center.append((y0 + y1) / 2)
            edge_text_center.append(f'{self.weights[(edge[0], edge[1])]}')

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.0, color='#888'), hoverinfo='none', mode='lines')
        weights_trace = go.Scatter(x = edge_x_center, y = edge_y_center, mode = 'text', marker_size = 0.5, text = edge_text_center, textposition = 'top center', hovertemplate = '<extra></extra>')

        node_x = []
        node_y = []
        for node in self.G.nodes():
            x, y = self.G.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            textposition='middle center',
            hovertemplate='<i>Bias:</i> %{customdata[0]}<br><i>Act_Fn:</i> %{customdata[1]}<br><i>Conn:</i> %{customdata[2]}<extra></extra>',
            marker=dict(
                showscale=True, 
                colorscale='armyrose',
                reversescale=True,
                color=[],
                size=25,
                line_width=2,
                colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right')))

        node_adjacencies = []
        node_text = []
        node_hover_data = []
        for node, adjacencies in enumerate(self.G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(f'{adjacencies[0]}')
            graph_node = self.G.nodes[adjacencies[0]]
            node_hover_data.append((graph_node['bias'], graph_node['act_fn'], list(adjacencies[1].keys())))

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text
        node_trace.customdata = node_hover_data

        #self.timestamp = time.strftime('%b-%d-%Y_%H%M%S', time.localtime())
        self.timestamp = time.time_ns()

        self.fig = go.Figure(data=[edge_trace, node_trace],#, weights_trace],
            layout=go.Layout(
                title=f'Neural Network Graph - {self.timestamp}',
                titlefont_size=14,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            )

    def show(self):
        self.fig.show()

    def save(self):
        filename = f'graph_visual/graph-{self.timestamp}.html'
        self.fig.write_html(filename)