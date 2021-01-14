## WVB Django

## API Details
Requests ('api') -- Changes pending

NOTE: All requests after login/register require a token=TOKEN in the header.

NOTE: args=body arguments, url or query string arguments are in the url.

NOTE: * required fields

-----------

User:

	'user/login/' POST
		args: username(string)*, password(string)*
		description: logs in an existing user, on validation returns authentication token (used in all subsequent requests)
	
	'user/register/' POST
		args: username(string)*, password(string)*, email(string)*
		description: register a new user, must be a unique username. Returns authentication token on success.
Project:

	'project/' GET
		args: None
		description: The list of all projects owned by the current authenticated users
	
	'project/' POST
		args: name(string)*, description(string), metadata(serialized json string)
		description: Creates a new project, metadata is used to populate the ProjectMetadata table
	
	'project/ID/' PUT
		args: name(string)*, description(string)
		description: Updates the parameters of the project, where the project ID.
	
	'project/ID/' DEL
		args: None
		description: Deletes the specified project, using the ID.
		
Dataset:

	'dataset/' GET
		args: None
		description: returns summaries of the datasets owned by the current authenticated user.
	
	'dataset/ID/' GET
		args: None
		description: returns the full dataset, if owned, by the provided dataset ID.
	
	'dataset/' POST
		args: data(csv)*, name(string)*, description(string), location_id(string), metadata(serialized json string)
		description: creates a dataset with the csv, metadata is a key:value pair dictionary, if response key not provided will be assigned to the name of the first column in the csv. Response should be determined by the front end UI.
	
	'dataset/ID/' PUT
		args: data(csv)*, name(string)*, description(string), location_id(string), metadata(serialized json string)
		description: updated existing dataset specified by ID.
	
	'dataset/ID/' Deletes
		args: None
		description: Deletes the specified dataset, using the ID.
		
Location:

	'location/' GET
		args: None
		description: returns a list of all locations owned by the user, from all projects.
	
	'location/' POST
		args: project_id(id)*, name(string)*, description(string), type(string)["None","Beach","Point",etc], metadata(serialized json string)
		description: Creates a new location for the specified project ID, metadata is a key:value pair dictionary where those entries correspond to the required parameters of the type. type=beach, metadata would need 3 lat/lng pairs.
	
	'location/ID/' PUT
		args: name(string)*, description(string), type(string), metadata(serialized json string)
		description: Updates an existing location, specified by the ID, with the details provided in the args.
	
	'location/ID' DEL
		args: None
		description: Deletes an existing location, specified by the ID.
		
AnalyticalModel:

	'analyticalmodel/project_id=ID' GET
		args: None
		description: return list of all analytical models for the specified project ID
	
	'analyticalmodel/' POST
		args: project_id(id)*, dataset_id(id), type(string)*, description(string), name(string)*, variables(json object), metadata(json object)
		description: creates a new analytical model object.
	
	'analyticalmodel/ID/' PUT
		args: project_id(id)*, dataset_id(id), type(string)*, description(string), name(string)*, variables(""), metadata("")
		description: updates an existing analytical model of the specified ID.
	
	'analyticalmodel/ID/' DEL
		args: None
		description: deletes an existing analytical model of the specified ID.