#include <stddef.h>

int read_memory(const void* pointer, size_t bytes) {
	int hash = 0;
	while (bytes >= 64) {
		hash ^= *((const int*) pointer);
		pointer += 64;
		bytes -= 64;		
	}
	return hash;
}
