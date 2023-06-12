import plotly.express as px
import plotly.graph_objects as go

from python_helpers.format_pdb import get_df

def plot_atoms(df, bounding_box=True):
    """plots atoms in 3D space using plotly"""
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='res_num')
    
    # adding bounding box
    if bounding_box:
        max_x, max_y, max_z = df[['x', 'y', 'z']].max()
        min_x, min_y, min_z = df[['x', 'y', 'z']].min()
        
        # creating bounding box (8 corners/points)
        corners = [[min_x, min_y, min_z], [min_x, min_y, max_z], 
                   [min_x, max_y, max_z], [min_x, max_y, min_z], 
                   
                   [max_x, max_y, min_z], [max_x, max_y, max_z],
                   [max_x, min_y, max_z], [max_x, min_y, min_z],
                   
                   # connecting corners
                   [max_x, max_y, min_z], [max_x, max_y, max_z],
                   [min_x, max_y, max_z], [min_x, max_y, min_z],
                   [min_x, min_y, min_z], [max_x, min_y, min_z],
                   [max_x, min_y, max_z], [min_x, min_y, max_z]]
        
        fig.add_trace(go.Scatter3d(x=[c[0] for c in corners],
                                   y=[c[1] for c in corners],
                                   z=[c[2] for c in corners]))
        
    fig.show()

def plot_together(dfL,dfP):
    """plotting ligand and protein in 3D space"""
    fig = px.scatter_3d(dfP, x='x', y='y', z='z', color='res_num')
    fig.add_scatter3d(x=dfL['x'], y=dfL['y'], z=dfL['z'], line={"color":'#000000'}, 
                      marker={"color":'#000000'}, name='ligand')
    fig.show()


if __name__ == '__main__':
    # split_structure(file_path='sample_data/1a1e_n-e.pdb', save='mains')
    dfP = get_df(open('sample_data/1a1e_n-e-split-1031.pdbqt','r').readlines())
    dfL = get_df(open('sample_data/1a1e_n-e-split-41_ligand.pdbqt','r').readlines())
    
    plot_together(dfL,dfP)