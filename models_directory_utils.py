def find_available_directory_name(path, base_name):
    i = 1
    while os.path.exists(os.path.join(path, f'{base_name}_{i}')):
        i += 1
    return f'{base_name}_{i}'

def create_directory(path, base_directory_name):
    directory_name = find_available_directory_name(path, base_directory_name)
    directory_path = os.path.join(path, directory_name)
    os.makedirs(directory_path)
    return directory_path

def write_hyperparameters(directory_path, parameters_dict):
    file_path = os.path.join(directory_path, 'hyperparameters.txt')
    with open(file_path, 'w') as file:
        for key, value in parameters_dict.items():
            file.write(f'{key}: {value}\n')
    print(f'Hyperparameter file created at path {file_path}')

def read_hyperparameters(file_path):
    hyperparameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            hyperparameters[key] = eval(value)
    return hyperparameters

def find_highest_numbered_directory(directory):
    highest_number = 0
    highest_directory = ''
    for directory_name in os.listdir(f'{directory}/.'):
            try:
                number = int(directory_name)
                if number > highest_number:
                    highest_number = number
                    highest_directory = directory_name
            except ValueError:
                # Ignore directories that don't match the expected pattern
                pass
    return os.path.join(directory, highest_directory), highest_number

def start():
    continue_training = False
    n_episodes_done = 0
    models_dir = 'Models'
    model_id = int(input('Id of model to train ?'))
    if os.path.exists(os.path.join(models_dir, f'Model_{model_id}')):
        model_dir_path = os.path.join(models_dir, f'Model_{model_id}')
        hp = read_hyperparameters(model_dir_path + '/hyperparameters.txt')
        choice = int(input('Create new model (1) or continue training (2) ?'))
        if choice == 1:
            training_dir_path = create_directory(model_dir_path, 'Training')
        elif choice == 2:
            training_id = int(input('Which training to continue ?'))
            training_dir_path = os.path.join(model_dir_path, 'Training_'+str(training_id))
            trained_dir_path,n_episodes_done = find_highest_numbered_directory(training_dir_path)
            if n_episodes_done != 0:
                continue_training = True
    else:
        model_dir_path = create_directory(models_dir, 'Model')
        write_hyperparameters(model_dir_path, hp)
        training_dir_path = os.path.join(model_dir_path, 'Training_1')
    return continue_training, trained_dir_path, training_dir_path, n_episodes_done