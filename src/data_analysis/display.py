import plotly.express as px
import plotly.graph_objects as go
from src.docking.python_helpers.format_pdb import get_atom_df

def get_corners(maxs, mins):
    """returns corners of bounding box"""
    return[ [mins[0], mins[1], mins[2]], [mins[0], mins[1], maxs[2]], 
            [mins[0], maxs[1], maxs[2]], [mins[0], maxs[1], mins[2]], 
            
            [maxs[0], maxs[1], mins[2]], [maxs[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]], [maxs[0], mins[1], mins[2]],
            
            # connecting corners
            [maxs[0], maxs[1], mins[2]], [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]], [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], mins[2]], [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]], [mins[0], mins[1], maxs[2]]]

def plot_atoms(df, bounding_box=True):
    """plots atoms in 3D space using plotly"""
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='res_num')
    
    # adding bounding box
    if bounding_box:
        corners = get_corners(df[['x', 'y', 'z']].max(), df[['x', 'y', 'z']].min())
        fig.add_trace(go.Scatter3d(x=[c[0] for c in corners],
                                y=[c[1] for c in corners],
                                z=[c[2] for c in corners]))
        
    fig.show()

def plot_together(dfL,dfP, show=True):
    """plotting ligand and protein in 3D space"""
    fig = px.scatter_3d(dfP, x='x', y='y', z='z', color='res_num')
    fig.add_scatter3d(x=dfL['x'], y=dfL['y'], z=dfL['z'], line={"color":'#000000'}, 
                      marker={"color":'#000000'}, name='ligand')
    if show:
        fig.show()
    return fig
        
def plot_all(vina_conf_path, show=True):
    """
    Similar to plot_together but with additional bounding box representing the pocket
    
    
    args:
        vina_conf_path: path to vina configuration file containing the pocket coordinates 
        and pdbqt file paths.
            example conf file:
                energy_range = 3
                exhaustiveness = 8
                num_modes = 9
                receptor = 1a1e/1a1e_protein.pdbqt
                ligand = 1a1e/1a1e_ligand.pdbqt
                out = 1a1e/1a1e_vina_out.pdbqt
                log = 1a1e/1a1e_vina_log.txt
                seed = 904455071
                center_x = 40.55822391857507
                center_y = -1.2030941475826973
                center_z = 39.733580152671756
                size_x = 26.237999999999996
                size_y = 16.667
                size_z = 34.822
    """
    # extracting info from vina conf file
    receptor_path = ''
    ligand_path = ''
    coords = {}
    with open(vina_conf_path, 'r') as f:
        for line in f.readlines():
            k,v = line.split('=')
            k = k.strip()
            v = v.strip()
            if k == 'receptor':
                receptor_path = v
            elif k == 'ligand':
                ligand_path = v
            elif k.startswith('center') or k.startswith('size'):
                coords[k] = float(v)
    
    assert receptor_path != '' and ligand_path != '', 'receptor and ligand paths not found'
    assert len(coords) == 6, 'center and/or size coordinates missing'
    
    # plotting receptor and ligand
    dfP = get_atom_df(open(ligand_path,'r').readlines())
    dfL = get_atom_df(open(receptor_path,'r').readlines())
    
    fig = plot_together(dfL,dfP, show=False)
    
    
    
    ###### TEMP TO SEE IF POCKET IS NOT DEFINED PROPERLY #########
    pocket_path = '/cluster/projects/kumargroup/jean/data/refined-set/1a1e/1a1e_pocket.pdb'
    pocket_df = get_atom_df(open(pocket_path,'r').readlines())
    
    coords["center_x"] = (pocket_df["x"].max() + pocket_df["x"].min()) / 2
    coords["center_y"] = (pocket_df["y"].max() + pocket_df["y"].min()) / 2
    coords["center_z"] = (pocket_df["z"].max() + pocket_df["z"].min()) / 2
    coords["size_x"] = pocket_df["x"].max() - pocket_df["x"].min()
    coords["size_y"] = pocket_df["y"].max() - pocket_df["y"].min()
    coords["size_z"] = pocket_df["z"].max() - pocket_df["z"].min()
    
    ###########################
    
    cx, cy, cz = coords['center_x'], coords['center_y'], coords['center_z']
    sx, sy, sz = coords['size_x']/2, coords['size_y']/2, coords['size_z']/2
    # adding bounding box
    fig.add_mesh3d(
        # 8 vertices of a cube
        x=[cx-sx, cx+sx, cx-sx, cx+sx,   cx-sx, cx+sx, cx-sx, cx+sx],
        y=[cy-sy, cy-sy, cy+sy, cy+sy,   cy-sy, cy-sy, cy+sy, cy+sy],
        z=[cz-sz, cz-sz, cz-sz, cz-sz,   cz+sz, cz+sz, cz+sz, cz+sz],

        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.5,
        color='#DC143C',
        flatshading = True)
    
    
    if show:
        fig.show()
    
    return fig

if __name__ == '__main__':
    dfP = get_atom_df(open('sample_data/1a1e_n-e-split-1031.pdbqt','r').readlines())
    dfL = get_atom_df(open('sample_data/1a1e_n-e-split-41_ligand.pdbqt','r').readlines())
    
    plot_together(dfL,dfP)