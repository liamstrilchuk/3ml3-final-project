from model import Model

# this function needs to be in a py file to create as a thread
# create, train, and run model, and return its characteristics
def evaluate_model(create_params={}, train_params={}):
	model = Model()
	model.create(**create_params)
	history, time_taken = model.train(**train_params)

	report = model.get_report(output_dict=True)
	return report, history, time_taken