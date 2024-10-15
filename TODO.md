Decided to backpedal on the template render_backend design for classes,
it solved the problem of virtual dispatch overhead, but introduced 
unnecessary complexity. 

To get the best of both worlds, we'll remove the templates and render_backend type,
and use compile flags to select between different backends, with ifndefs and such.