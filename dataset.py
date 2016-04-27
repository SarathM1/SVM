from sklearn.datasets import fetch_species_distributions 

# Extracting Data from web
data = fetch_species_distributions()

# Print all the subsection available

print "keys = ", data.keys()
print "\n\n Hit Enter to Continue"
raw_input()

print "Data = "
print "_"*80
print data
print "_"*80
