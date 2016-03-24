#include <nacl/interfaces.h>
#include <nacl/stringvars.h>

static PP_Bool on_create_instance(PP_Instance instance, uint32_t argc, const char* argn[], const char* argv[]) {
	return PP_TRUE;
}

static void on_destroy_instance(PP_Instance instance) {
}

static void on_change_view(PP_Instance instance, PP_Resource view) {
}

static void on_change_focus(PP_Instance instance, PP_Bool has_focus) {
}

static PP_Bool on_document_load(PP_Instance instance, PP_Resource url_loader) {
	return PP_FALSE;
}

const struct PPP_Instance_1_1 plugin_instance_interface = {
	.DidCreate = on_create_instance,
	.DidDestroy = on_destroy_instance,
	.DidChangeView = on_change_view,
	.DidChangeFocus = on_change_focus,
	.HandleDocumentLoad = on_document_load
};

