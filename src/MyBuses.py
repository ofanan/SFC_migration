from doopl.factory import *

# Data

X=[
    (1, 1, 40,500),
    (2, 1, 30,400)
    ]

# Create an OPL model from a .mod file
with create_opl_model(model="MyBuses.mod") as opl:
    # tuple can be a list of tuples, a pandas dataframe...
    opl.set_input("buses", X)

    # Generate the problem and solve it.
    opl.run()

    # Get the names of post processing tables
    print("Table names are: "+ str(opl.output_table_names))

    # Get all the post processing tables as dataframes.
    for name, table in iteritems(opl.report):
        #print("Table : " + name)
        #for t in table.itertuples(index=False):
        #    print(t)

        # nicer display
        for t in table.itertuples(index=False):
            print("id", t[1],": ", t[0])