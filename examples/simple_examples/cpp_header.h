#ifndef LibraryX
#define LibraryX

#ifdef __cplusplus
extern "C" {
#endif

// // Declare the type of the function pointer you are returning
// typedef void (*GeneratedFuncType)(void*, void*, void*);

// Declare the generateCode function
// GeneratedFuncType
void(*generateCode(const char* script))(void);

#ifdef  __cplusplus
}
#endif

#endif