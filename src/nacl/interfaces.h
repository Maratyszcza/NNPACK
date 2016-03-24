#include <ppapi/c/ppb_core.h>
#include <ppapi/c/ppb_console.h>
#include <ppapi/c/ppb_messaging.h>
#include <ppapi/c/ppb_var.h>
#include <ppapi/c/ppb_var_dictionary.h>

#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/ppp_messaging.h>

extern const struct PPB_Core_1_0* core_interface;
extern const struct PPB_Console_1_0* console_interface;
extern const struct PPB_Messaging_1_0* messaging_interface;
extern const struct PPB_Var_1_1* var_interface;
extern const struct PPB_VarDictionary_1_0* dictionary_interface;

extern const struct PPP_Messaging_1_0 plugin_messaging_interface;
extern const struct PPP_Instance_1_1 plugin_instance_interface;

