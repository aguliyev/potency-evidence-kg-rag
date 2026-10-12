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

----------------

you have a table for `alias_candidate`, 
but probably it should be `aliases` with a status pending/approved/rejected, 
so that candidates can be easily made functioning aliases.
and you have some aliases in python code in project_profiles.py ,  let us make it a json file instead, 
with aliases we know for all tenants/projects, and it be loaded into the `aliases` table.

also, suggest if we can use a "corpus-wide" (probably per document) pass to identify more aliases, and load them into the `aliases` table for review.
This could be with LLM, but also some deterministic algorithms?



