from src.main.python.catalog.Catalog import get_data_catalog

if __name__ == '__main__':
    cat = get_data_catalog(dataset_name='data/adult.dat', file_format='csv')
    a = 100
