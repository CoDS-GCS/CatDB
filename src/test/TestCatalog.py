import src.main.python.catalog.Catalog as catalog



if __name__ == '__main__':
    cat = catalog.Catalog(dataset_name='data/adult.dat')
    cat.profile_dataset()