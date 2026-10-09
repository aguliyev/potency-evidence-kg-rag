Agents stop: This file is for humans only.


------------------------


Make Docker Compose and two Docker files. 
One Docker container will be Jupiter notebook. 
Another Docker container will be Postgress database 
with PG vector and Apache Age, which allows for a graph database.

make bin/ with Shell Scripps to `install`, `start` and `stop` our system.
And make a lib/ directory for code which will be used in the notebook. 

this lib/ and notebooks/ should be mapped local directories to the Jupiter container 
so that I can change the code and Jupiter sees the updated files.

Starting the Postgres container should also run migrations. Which Python package do you suggest for that? Should we use SQL alchemy and alambic or something else? 