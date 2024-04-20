import tkinter as tk

def select_option(value, row):
    global selected_values
    selected_values[row] = value
    # Update button colors in the row
    for val, btn in button_sets[row].items():
        if val == value:
            btn.config(bg='darkgrey')  # Highlight the selected button
        else:
            btn.config(bg=original_colors[btn])  # Reset to original color
    # Check if selections have been made in both rows

    if all(row in selected_values for row in button_sets):
        ok_button.config(state='normal')  # Enable the OK button
    else:
        ok_button.config(state='disabled')  # Keep the OK button disabled

def ok():
    selector_tk.destroy()  # Close the window after selection

def selector():
    global selector_tk, button_sets, selected_values, original_colors, ok_button
    selector_tk = tk.Tk()
    selector_tk.title("ODM Option Selector")
    selector_tk.geometry('380x200')

    button_sets = {}
    selected_values = {}
    original_colors = {}

    # Top frame for the first row of buttons
    top_frame = tk.Frame(selector_tk)
    top_frame.grid(row=0, column=0, columnspan=3, sticky='ew')
    selector_tk.grid_rowconfigure(0, weight=1)
    top_frame.grid_columnconfigure(0, weight=1)
    top_frame.grid_columnconfigure(1, weight=1)

    # First row buttons
    envs = ['Inverted Pendulum', 'Double Inverted Pendulum']
    button_sets[0] = {}
    for index, value in enumerate(envs):
        button = tk.Button(top_frame, text=f'{value}', command=lambda v=value, r=0: select_option(v, r))
        button.grid(row=0, column=index, padx=10, pady=10, sticky='ew')
        button_sets[0][value] = button
        original_colors[button] = button.cget('background')

    # Bottom frame for the second row of buttons
    bottom_frame = tk.Frame(selector_tk)
    bottom_frame.grid(row=1, column=0, columnspan=3, sticky='ew')
    selector_tk.grid_rowconfigure(1, weight=1)
    for i in range(3):
        bottom_frame.grid_columnconfigure(i, weight=1)

    # Second row buttons
    values_row2 = ['FQI', 'REINFORCE', 'PPO']
    button_sets[1] = {}
    for index, value in enumerate(values_row2):
        button = tk.Button(bottom_frame, text=f'{value}', command=lambda v=value, r=1: select_option(v, r))
        button.grid(row=0, column=index, padx=10, pady=10, sticky='ew')
        button_sets[1][value] = button
        original_colors[button] = button.cget('background')

    # OK button to confirm the selection
    ok_button = tk.Button(selector_tk, text='OK', command=ok, state='disabled')
    ok_button.grid(row=2, column=1, padx=10, pady=20, sticky='ew')
    selector_tk.grid_rowconfigure(2, weight=1)

    selector_tk.mainloop()

    return selected_values
