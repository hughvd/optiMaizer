class Method:
    def __init__(self, name, **options):
        self.name = name
        self.options = options


class Options:
    def __init__(self, term_tol=1e-6, max_iterations=1e2):
        self.term_tol = term_tol
        self.max_iterations = max_iterations
