from .Adult import Adult

def init_data_class(dest_folder, batch_size, seed=42):
    return Adult(dest_folder = dest_folder, batch_size=batch_size, seed=seed)
