from model import Model

def evaluate_model(create_params={}, train_params={}):
	model = Model()
	model.create(**create_params)
	_, time_taken = model.train(**train_params)

	report = model.get_report(output_dict=True)
	return report, time_taken