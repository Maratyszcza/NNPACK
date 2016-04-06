#include <stddef.h>
#include <string.h>

#include <ppapi/c/pp_bool.h>
#include <ppapi/c/pp_errors.h>
#include <ppapi/c/ppp.h>

#include <nacl/interfaces.h>
#include <nacl/stringvars.h>

#include <nnpack.h>

PP_EXPORT int32_t PPP_InitializeModule(PP_Module module, PPB_GetInterface get_browser_interface) {
	core_interface = get_browser_interface(PPB_CORE_INTERFACE_1_0);
	if (core_interface == NULL) {
		return PP_ERROR_NOINTERFACE;
	}
	console_interface = get_browser_interface(PPB_CONSOLE_INTERFACE_1_0);
	if (console_interface == NULL) {
		return PP_ERROR_NOINTERFACE;
	}
	messaging_interface = get_browser_interface(PPB_MESSAGING_INTERFACE_1_0);
	if (messaging_interface == NULL) {
		return PP_ERROR_NOINTERFACE;
	}
	var_interface = get_browser_interface(PPB_VAR_INTERFACE_1_1);
	if (var_interface == NULL) {
		return PP_ERROR_NOINTERFACE;
	}
	dictionary_interface = get_browser_interface(PPB_VAR_DICTIONARY_INTERFACE_1_0);
	if (dictionary_interface == NULL) {
		return PP_ERROR_NOINTERFACE;
	}
	init_string_vars();

	enum nnp_status status = nnp_initialize();
	if (status != nnp_status_success) {
		return PP_ERROR_FAILED;
	}

	return PP_OK;
}

PP_EXPORT void PPP_ShutdownModule(void) {
	nnp_deinitialize();

	release_string_vars();
}

PP_EXPORT const void* PPP_GetInterface(const char* interface_name) {
	if (strcmp(interface_name, PPP_INSTANCE_INTERFACE_1_1) == 0) {
		return &plugin_instance_interface;
	} else if (strcmp(interface_name, PPP_MESSAGING_INTERFACE_1_0) == 0) {
		return &plugin_messaging_interface;
	} else {
		return NULL;
	}
}

