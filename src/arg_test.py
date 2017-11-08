import argparse

parser = argparse.ArgumentParser(description="Embed complex network to hyperbolic space.")
parser.add_argument("-A", "--gml_file", dest="gml_file", 
                    help="path of gml file of network")
parser.add_argument("-X", "--attribute_file", dest="attribute_file", 
                    help="path of attribute file")

args = parser.parse_args()

print args.gml_file
print args.attribute_file

