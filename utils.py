import streamlit as st

# tag::write_message[]
def write_message(role, content, save = True, isError = False):
    """
    This is a helper function that saves a message to the
     session state and then writes a message to the UI
    """
    # Append to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content, "isError": isError})

    # Write to UI
    with st.chat_message(role):
        if isError:
            st.markdown("ERROR! The following error was encountered:")
            st.error(content)
        else:
            st.markdown(content)
# end::write_message[]
