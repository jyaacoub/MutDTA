from src.models.helpers.contact_map import get_contact

code = '1a1e'
path = f'/cluster/projects/kumargroup/jean/data/refined-set/{code}/{code}_protein.pdb'
get_contact(path, display=True)
