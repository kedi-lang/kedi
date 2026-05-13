// Python binding for the tree-sitter-kedi grammar.
//
// Exposes a single function `language()` that returns a PyCapsule
// wrapping the `TSLanguage*` produced by `tree_sitter_kedi()` (the
// symbol defined in the generated parser.c). The capsule is consumed
// by `tree_sitter.Language(capsule)` on the Python side.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct TSLanguage TSLanguage;

extern const TSLanguage *tree_sitter_kedi(void);

static PyObject *_binding_language(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyCapsule_New((void *)tree_sitter_kedi(), "tree_sitter.Language", NULL);
}

static PyMethodDef methods[] = {
    {
        "language",
        _binding_language,
        METH_NOARGS,
        "Get the tree-sitter language pointer for Kedi as a PyCapsule.",
    },
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_binding",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__binding(void) {
    return PyModule_Create(&module_def);
}
