"""
We notice that opengeos (the detection part)-> https://github.com/opengeos/segment-geospatial
if simply a Grounding-Dino with different Config file and Checkpoint pth file,
the Pth file is -> https://github.com/opengeos/segment-geospatial : groundingdino_swinb_cogcoor.pth
The config is -> https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB_cfg.py :
which is -> groundingdino/config/GroundingDINO_SwinB_cfg.py
so we actually dont need to write any code for that in this scope of the project, simply used the
../GroundindDINO_workspace
with the config and the pth file we mentioned :)
"""