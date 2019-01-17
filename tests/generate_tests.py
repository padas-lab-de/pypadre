from padre.visitors.mappings import type_mappings

def test_other_test():
    from padre.visitors.mappings import type_mappings
    for k in type_mappings:
        if k:
            print("def test_extract_" + type_mappings[k][0]['name'].replace(' ', '_').replace('-', '_') + "(self): ")
            for impl in type_mappings[k][0]['implementation']:
                # impl = type_mappings[k][0]['implementation']['scikit-learn']
                print("    self.extract_type_object(" + impl + ")")
                print()


if __name__=='__main__':
    test_other_test()
