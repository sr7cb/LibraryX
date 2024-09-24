#ifndef LibraryX
#define LibraryX


#ifdef __cplusplus
extern "C" {
#endif

float (Executor::* generateDynamicCode(const char* script))(void*&);

#ifdef  __cplusplus
}
#endif

#endif