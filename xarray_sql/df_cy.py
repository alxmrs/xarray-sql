# distutils: language = c++
import cython
# #ifndef ARROW_C_DATA_INTERFACE
# #define ARROW_C_DATA_INTERFACE
#
# #define ARROW_FLAG_DICTIONARY_ORDERED 1
# #define ARROW_FLAG_NULLABLE 2
# #define ARROW_FLAG_MAP_KEYS_SORTED 4
#
# struct ArrowSchema {
#   // Array type description
#   const char* format;
#   const char* name;
#   const char* metadata;
#   int64_t flags;
#   int64_t n_children;
#   struct ArrowSchema** children;
#   struct ArrowSchema* dictionary;
#
#   // Release callback
#   void (*release)(struct ArrowSchema*);
#   // Opaque producer-specific data
#   void* private_data;
# };
#
# struct ArrowArray {
#   // Array data description
#   int64_t length;
#   int64_t null_count;
#   int64_t offset;
#   int64_t n_buffers;
#   int64_t n_children;
#   const void** buffers;
#   struct ArrowArray** children;
#   struct ArrowArray* dictionary;
#
#   // Release callback
#   void (*release)(struct ArrowArray*);
#   // Opaque producer-specific data
#   void* private_data;
# };
#
# #endif  // ARROW_C_DATA_INTERFACE


# #ifndef ARROW_C_STREAM_INTERFACE
# #define ARROW_C_STREAM_INTERFACE
#
# struct ArrowArrayStream {
#   // Callbacks providing stream functionality
#   int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
#   int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);
#   const char* (*get_last_error)(struct ArrowArrayStream*);
#
#   // Release callback
#   void (*release)(struct ArrowArrayStream*);
#
#   // Opaque producer-specific data
#   void* private_data;
# };
#
# #endif  // ARROW_C_STREAM_INTERFACE


# cimport cpython
# from libc.stdlib cimport malloc, free
#
# cdef void release_arrow_schema_py_capsule(object schema_capsule):
#     cdef ArrowSchema* schema = <ArrowSchema*>cpython.PyCapsule_GetPointer(
#         schema_capsule, 'arrow_schema'
#     )
#     if schema.release != NULL:
#         schema.release(schema)
#
#     free(schema)
#
# cdef object export_arrow_schema_py_capsule():
#     cdef ArrowSchema* schema = <ArrowSchema*>malloc(sizeof(ArrowSchema))
#     # It's recommended to immediately wrap the struct in a capsule, so
#     # if subsequent lines raise an exception memory will not be leaked.
#     schema.release = NULL
#     capsule = cpython.PyCapsule_New(
#         <void*>schema, 'arrow_schema', release_arrow_schema_py_capsule
#     )
#     # Fill in ArrowSchema fields:
#     # schema.format = ...
#     # ...
#     return capsule

from cython.cimports import pyarrow as pa
from cython.cimports.libc.stdlib import malloc, free

@cython.cclass
class XarrayDatasetTable:

  def __init__(self):
    pass

  def __arrow_c_stream__(self, requested_schema: None = None) -> object:
    # requested_schema is primarily used to cast the data types of the final
    # schema, not to do column agreement.
    return None
    # A PyCapsule containing a C ArrowArrayStream representation of the
    # object. The capsule must have a name of "arrow_array_stream".

    # It's recommended to immediately wrap the struct in a capsule, so
    # if subsequent lines raise an exception memory will not be leaked.
    # schema.release = NULL
    # capsule = cpython.PyCapsule_New(
    #           < void * > schema, 'arrow_schema', release_arrow_schema_py_capsule
    # )
    # Fill in ArrowSchema fields:
    # schema.format = ...
    # ...
    # return capsule

