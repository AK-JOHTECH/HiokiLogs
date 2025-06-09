import tkinter.ttk as ttk
import re
import csv
import pandas as pd
import math
import tkinter as tk
import customtkinter # <- import the CustomTkinter module
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import tkinter.filedialog as fd
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('blue')


def end_program():
    #********************************************************************************************
    #Description: This function shuts down the main window root. 
    #             Input: Void. 
    #             Output: Void.  
    #********************************************************************************************
    plt.close()
    plt.close()
    plt.close()
    root.destroy()
    
    
#====================================================================

#===============Window Class Object==================================
class Window(customtkinter.CTkToplevel): 
    #********************************************************************************************
    #Description: This class is used to create the main windows and connecting windows. 
        # It connects the main root to all the sub windows. It also sets a default size 
        # for a window, but size can be changed using sub roots. 
    # Functions: None
    #********************************************************************************************
    def __init__(self):
        super().__init__()
        self.protocol('WM_DELETE_WINDOW',end_program)
#====================================================================
#===============Pop Up Class Object==================================
class PopUpWindow(customtkinter.CTkToplevel): 
    #********************************************************************************************
    #Description: This class is used to create all the pop up windows.
    # Functions: content 
  
    #********************************************************************************************
    def __init__(self):
        super().__init__()
        self.title('Error')
        self.geometry('600x150')
 

    def content(self,Text):
        #********************************************************************************************
        #Description: (public) This function displays the error message. All the attributes like font size,
        #   color, and close button can be  set here.
        #             Input: Error message. 
        #             Output: Void.  
        #********************************************************************************************
        error_message = customtkinter.CTkLabel(self, text = Text) 
        error_message.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        def close_popup():
        #********************************************************************************************
        #Description: This function shuts down the pop up window root. 
        #             Input: Void. 
        #             Output: Void.  
        #********************************************************************************************
            self.destroy()

        button = customtkinter.CTkButton(self, text = 'OK',command = close_popup,fg_color = 'green',hover_color='#34eb7a',corner_radius = 80)
        button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
        self.grab_set()  # Prevent interaction with the main window
        self.wait_window()  # Wait until the pop-up window is closed
        self.protocol('WM_DELETE_WINDOW',close_popup)
#====================================================================

#====================================================================
# Function to extract the numeric part and sort the file paths
def sort_file_paths(file_paths):
    # Define the pattern to extract the last numeric part after the last '_'
    pattern = re.compile(r'_(\d{4})\.CSV$', re.IGNORECASE)
    
    # Sort the file paths based on the extracted numeric part
    sorted_file_paths = sorted(file_paths, key=lambda x: int(pattern.search(x).group(1)))
    
    return sorted_file_paths

#====================================================================
def combine_csv_files(csv_files):
    # Define the desired headers for the final CSV file
    row_number = 6  # Extract the 10th row (assuming 1-based row index)
    df = pd.read_csv(csv_files[0], skiprows=row_number, nrows=1)
    # Convert the DataFrame row to a list
    global headers
    headers = df.values.flatten().tolist() 
    headers[0] = 'Time'
    headers.append('Event')
    print(headers)
    dataframe = pd.DataFrame(columns=headers)  # Empty DataFrame to store all data

    for file in csv_files:
        df = pd.read_csv(file, skiprows=12)
        # # Concatenate with ignore_index=True to reset the index in the final DataFrame
        df.columns = headers[:len(df.columns)]  # This forces the columns to match the headers' length
        dataframe = pd.concat([dataframe, df], ignore_index=True)
        
    # dataframe.to_csv(output_file, header=headers, index=False)

    return dataframe
#====================================================================
#====================================================================
def filter_dataframe(filters):
    global dataframe
    # Extract columns where the filter value is 1
    columns_to_extract = ['Time'] + [col for col, value in filters.items() if value == 1] + ['Event']

    # Create a new DataFrame with only the selected columns
    new_dataframe = dataframe[columns_to_extract]
    return new_dataframe
#====================================================================
#====================================================================
def main_window():
    #********************************************************************************************
    #Description: This is the main function. It contains the slave function for 
    #   file browsing and error checks. 
    #             Input: Void 
    #             Output: Void.  
    #********************************************************************************************
    global main_root
    main_root = Window()
    main_root.geometry('305x500')
    main_root.title('Johanson MDF Reader')

    frame_1 = customtkinter.CTkFrame(main_root,width = 300,height = 500,corner_radius=20,fg_color='#2e353e')
    frame_1.place(relx=0.01, rely=0.01)

    frame_image = customtkinter.CTkImage(dark_image=Image.open(r"H:\Johanson Enginering Programs\Tyler's Project\Logo\logo.png"),size=(150, 150))
    image_label = customtkinter.CTkLabel(frame_1, image=frame_image, text="")
    image_label.place(relx=0.25,rely=0.04)

    main_path_loc = customtkinter.CTkEntry(frame_1,placeholder_text="CSV Files",corner_radius=20,width = 280,height = 20)
    main_path_loc.place(relx=0.03,rely=0.45)
   
    # Function to handle the file browse action
    def browse_files():
        global file_paths
        file_paths = []
        file_paths = fd.askopenfilenames(title="Browse Files", filetypes=[('CSV Files','*.CSV')])
        #("MF4 Files", "*.MF4")
        if file_paths:
            selected_files = "\n".join(file_paths)
        main_path_loc.insert(0,selected_files)
        print('selected files',selected_files)
        global sorted_file_paths
        sorted_file_paths = sort_file_paths(file_paths)
        
    def compile():
        try:
            # output_file = r"C:\Users\akohli\Music\Tyler's Project\Output.CSV"  # Path to save the output CSV
            global sorted_file_paths
            global dataframe
            dataframe = combine_csv_files(sorted_file_paths)
            plotter(dataframe)
            # controls(file_paths)
            main_root.geometry('1300x1000')
        except IndexError:
            pop_up_root = PopUpWindow()
            text = "Please select the CSV Files." 
            PopUpWindow.content(pop_up_root,text)
        except Exception as e:
            pop_up_root = PopUpWindow()
            text = f"Please select the CSV Files.\n{e}" 
            PopUpWindow.content(pop_up_root,text)
        

    browse = customtkinter.CTkButton(frame_1,text = 'Browse',corner_radius=20,command = browse_files)
    browse.place(relx=0.25,rely=0.6)

    button = customtkinter.CTkButton(frame_1,text = 'Compile',corner_radius=20,command = compile)
    button.place(relx=0.25,rely=0.75)

def plotter(dataframe, previous_checkbox_values=None):
    global x_lab, y_lab, plt_marker, plt_line
    plt_line = '-'
    plt_marker = 'o'
    x_lab = 'Time (s)'
    y_lab = 'Temperature (\u2103)'

    # Plot Frame
    frame_2 = customtkinter.CTkFrame(main_root, width=670, height=500, corner_radius=20, fg_color='#2e353e')
    frame_2.place(relx=0.25, rely=0.01)

    # Plot controls Frame
    frame_3 = customtkinter.CTkFrame(main_root, width=290, height=500, corner_radius=20, fg_color='#2e353e')
    frame_3.place(relx=0.77, rely=0.01)

    # Label Frame
    frame_4 = customtkinter.CTkFrame(main_root, width=300, height=400, corner_radius=20, fg_color='#2e353e')
    frame_4.place(relx=0.01, rely=0.53)

    # Label Frame
    global frame_5
    frame_5 = customtkinter.CTkFrame(main_root, width=290, height=400, corner_radius=20, fg_color='#2e353e')
    frame_5.place(relx=0.77, rely=0.53)
    control_frame_5(previous_checkbox_values)

    # Label Frame
    frame_6 = customtkinter.CTkFrame(main_root, width=670, height=400, corner_radius=20, fg_color='#2e353e')
    frame_6.place(relx=0.25, rely=0.53)
    # Pass previous checkbox values to control_frame_6 to retain checkbox states
    

    # Create the initial plot with multiple lines
    global plt
    fig, ax = plt.subplots()
    ax.set_facecolor('#2e353e')
    fig.set_facecolor('#2e353e')

    # Store line objects in a dictionary to update later
    lines = {}

    # Read and plot each relevant column from the dataframe
    for i, col in enumerate(dataframe.columns):
        if col == "Time" or col == "Event":
            continue  # Skip the "Time" and "Event" columns

        x = dataframe["Time"]
        y = dataframe[col]
        line, = ax.plot(x, y, marker=plt_marker, linestyle=plt_line, label=f"{col}")
        lines[i] = line

    plt.xlabel(x_lab, color='white')
    plt.ylabel(y_lab, color='white')
   
    # Customize the plot aesthetics
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Embed the plot in the frame
    canvas = FigureCanvasTkAgg(fig, master=frame_2)
    canvas.draw()
    canvas.get_tk_widget().place(relx=0.0195, rely=0.01)

    # Variables to store slider values
    slider_min_val = customtkinter.IntVar(value=0)
    slider_max_val = customtkinter.IntVar(value=100)
#==================================================================================
    def update_plot(_):
        for line in lines.values():
            line.set_marker(plt_marker)
            line.set_linestyle(plt_line)

        # Get the slider values for the time range
        min_val = slider_min_val.get() / 100 * (dataframe["Time"].max() - dataframe["Time"].min()) + dataframe["Time"].min()
        max_val = slider_max_val.get() / 100 * (dataframe["Time"].max() - dataframe["Time"].min()) + dataframe["Time"].min()

        # Trigger an error if max_val is lower than min_val
        if max_val <= min_val:
            min_box.delete(0, customtkinter.END)
            min_box.insert(0, "Error!")
            max_box.delete(0, customtkinter.END)
            max_box.insert(0, "Error!")
            return

        # Update the entries
        min_box.delete(0, customtkinter.END)
        min_box.insert(0, f"{min_val:.2f}")
        max_box.delete(0, customtkinter.END)
        max_box.insert(0, f"{max_val:.2f}")

        # Set the new X-axis limits based on the slider values
        ax.set_xlim(min_val, max_val)

        # Update overall statistics and the table
        table.delete(*table.get_children())  # Clear existing table entries
        overall_min_y, overall_max_y = float('inf'), float('-inf')  # Track global y-limits
        if check == 1:
            plt.legend()
        else:
            legend = ax.get_legend()  # Get the legend object from the plot
            if legend:
                legend.remove()  # Remove the legend if it exists
        
        plt.xlabel(x_lab, color='white')
        plt.ylabel(y_lab, color='white')

        # Iterate through the lines dictionary and use column names to get the correct data
        for col_name, line in zip(dataframe.columns[1:], lines.values()):  # Skip "Time" column
            x = dataframe["Time"]
            y = dataframe[col_name]  # Use the correct column name for y data
            
            # Apply the slider limits
            start_idx = np.argmin(np.abs(np.array(x) - min_val))
            end_idx = np.argmin(np.abs(np.array(x) - max_val))

            new_x = x[start_idx:end_idx + 1]
            new_y = y[start_idx:end_idx + 1]
            line.set_data(new_x, new_y)

            # Update overall y limits
            overall_min_y = min(overall_min_y, np.min(new_y))
            overall_max_y = max(overall_max_y, np.max(new_y))

            # Update statistics
            stdev = np.std(new_y)
            variance = np.var(new_y)
            rnge = np.max(new_y) - np.min(new_y)
            slope = (new_y.iloc[-1] - new_y.iloc[0]) / (new_x.iloc[-1] - new_x.iloc[0]) if len(new_x) > 1 else 0

            # Insert statistics into the table
            table.insert('', 'end', values=(f'{col_name}', f'{slope:.2f}', f'{rnge:.2f}', f'{stdev:.2f}', f'{variance:.2f}'))

        ax.set_ylim(overall_min_y, overall_max_y)
        fig.canvas.draw()
#==================================================================================
    # Sliders to adjust the X-axis limits
    slider_min = customtkinter.CTkSlider(frame_3, from_=0, to=100, command=update_plot, variable=slider_min_val, orientation='vertical', height=350, width=25)
    slider_min.place(relx=0.25, rely=0.1)

    slider_max = customtkinter.CTkSlider(frame_3, from_=0, to=100, command=update_plot, variable=slider_max_val, orientation='vertical', height=350, width=25)
    slider_max.place(relx=0.72, rely=0.1)

    # Min Slider value box
    min_box = customtkinter.CTkEntry(frame_3, placeholder_text="Min Val.", corner_radius=20, width=80, height=30)
    min_box.place(relx=0.17, rely=0.85)

    # Max Slider value box
    max_box = customtkinter.CTkEntry(frame_3, placeholder_text="Max Val.", corner_radius=20, width=80, height=30)
    max_box.place(relx=0.63, rely=0.85)

    # X-label entry
    customtkinter.CTkLabel(frame_4, text='X-Label:', font=('Helvetica', 14)).place(relx=0.09, rely=0.2)
    x_label = customtkinter.CTkEntry(frame_4, placeholder_text="X-Label", corner_radius=20, width=150, height=30)
    x_label.place(relx=0.28, rely=0.2)
    x_label.insert(0, x_lab)

    # Y-label entry
    customtkinter.CTkLabel(frame_4, text='Y-Label:', font=('Helvetica', 14)).place(relx=0.09, rely=0.4)
    y_label = customtkinter.CTkEntry(frame_4, placeholder_text="Y-Label", corner_radius=20, width=150, height=30)
    y_label.place(relx=0.28, rely=0.4)
    y_label.insert(0, y_lab)
    
    # Plot Button function
    def plot(_=None):
        global x_lab, y_lab, check, plt_marker, plt_line
        x_lab = x_label.get()
        y_lab = y_label.get()
        check = check_var.get()
        plt_marker = dropdown_marker.get()
        plt_line = dropdown_line.get()
        update_plot(None)

    
    # Legends
    check_var = customtkinter.IntVar(value=0)
    customtkinter.CTkCheckBox(frame_4, text='Legends', onvalue=1, offvalue=0, variable=check_var, font=('Helvetica', 14, 'bold')).place(relx=0.09, rely=0.6)

    # Marker style options
    customtkinter.CTkLabel(frame_4, text='Marker Options:', font=('Helvetica', 14)).place(relx=0.09, rely=0.7)
    marker_options = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_', 'None']
    dropdown_marker = customtkinter.CTkOptionMenu(frame_4, values=marker_options, command=update_plot)
    dropdown_marker.place(relx=0.5, rely=0.7)

    # Line style options
    customtkinter.CTkLabel(frame_4, text='Line Options:', font=('Helvetica', 14)).place(relx=0.09, rely=0.8)
    line_options = ['-', '--', '-.', ':', 'None', 'solid', 'dashed', 'dashdot', 'dotted']
    dropdown_line = customtkinter.CTkOptionMenu(frame_4, values=line_options, command=update_plot)
    dropdown_line.place(relx=0.5, rely=0.8)


    # Statistics Table
    # Table for Statistics
    style = ttk.Style()

    # Set the theme to 'clam' to allow customization
    style.theme_use('clam')
   # Customize the Treeview background, foreground, and grid colors
    style.configure("Treeview", 
                    background="#2e353e",     # Background color for rows
                    foreground="white",        # Text color for rows
                    fieldbackground="#2e353e", # Background color for entry fields
                    bordercolor="white",       # Border color of the table/grid
                    borderwidth=1)

    # Customize the heading colors
    style.configure("Treeview.Heading", 
                    background="#4f5b66",      # Background color of the heading
                    foreground="white",        # Text color of the heading
                    borderwidth=1)

    # Add a grid to the table
    style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])
    customtkinter.CTkLabel(frame_6, text='Statistics', font=('Helvetica', 30)).place(relx=0.30, rely=0.015)
    # Apply the style to the table
    table = ttk.Treeview(frame_6, 
                        columns=('Dataset','Slope','Range', 'St.Dev', 'Variance'), 
                        show='headings', 
                        style="Treeview")

    table = ttk.Treeview(frame_6, columns=('Dataset','Slope','Range', 'St.Dev', 'Variance'), show='headings')
    table.heading('Dataset', text='Dataset')
    table.heading('Range', text='Range')
    table.heading('St.Dev', text='\u03C3')
    table.heading('Variance', text=f'\u03C3^2')
    table.heading('Slope', text='Slope')
    table.column('Dataset', width=50)
    table.column('Range', width=50)
    table.column('St.Dev', width=50)
    table.column('Variance', width=50)
    table.column('Slope', width=50)
    table.place(relx=0.035, rely=0.15, relwidth=0.95, relheight=0.80)

    plot_button = customtkinter.CTkButton(frame_3, text='Plot', corner_radius=10, command=plot)
    plot_button.place(relx=0.28, rely=0.02)

def control_frame_5(previous_checkbox_values=None):
    global frame_5
    global headers
    new_headers = headers.copy()
    new_headers.pop(0)  # Removing first element (e.g., 'Time')
    new_headers.pop()   # Removing last element (e.g., 'Event')

    # A dictionary to hold the IntVar variables for each checkbox
    checkbox_vars = {}

    # Create a label for the Channel Selector
    customtkinter.CTkLabel(frame_5, text='Channel Selector', font=('Helvetica', 20)).place(relx=0.28, rely=0.06)

    # Positioning variables
    relx_base = 0.03  # Base x position
    rely_base = 0.15  # Base y position
    relx_step = 0.32 # Horizontal step between checkboxes
    rely_step = 0.10  # Vertical step between rows

    # Dynamically create checkboxes based on the headers
    for i, header in enumerate(new_headers):
        # Restore previous checkbox state if available, otherwise set to 0 (unchecked)
        initial_value = previous_checkbox_values.get(header, 0) if previous_checkbox_values else 0

        # Create an IntVar for each checkbox with the preserved state
        checkbox_vars[header] = customtkinter.IntVar(value=initial_value)

        # Calculate position of the checkbox
        relx = relx_base + (i % 3) * relx_step  # 6 checkboxes per row
        rely = rely_base + (i // 3) * rely_step  # Move to the next row after 6 columns

        # Create the checkbox with dynamic text and variable
        customtkinter.CTkCheckBox(frame_5, text=header, onvalue=1, offvalue=0, variable=checkbox_vars[header], font=('Helvetica', 12, 'bold')).place(relx=relx, rely=rely)

    # Function to get the current state of all checkboxes
    def get_checkbox_values():
        filters = {header: checkbox_vars[header].get() for header in new_headers}
        new_dataframe = filter_dataframe(filters)

        # Store current checkbox states
        previous_checkbox_values = filters.copy()  # Store the states before replotting
        plt.close()
        plotter(new_dataframe, previous_checkbox_values)  # Pass current checkbox states to the plotter function
        

    # Example button to trigger the output of checkbox values
    customtkinter.CTkButton(frame_5, text="Re-Plot", command=get_checkbox_values).place(relx=0.28, rely=0.85)


def PID():
    import control as ctrl

    # Define the parameters of the FOPDT model
    K = 1.0  # process gain
    tau = 5.0  # time constant
    theta = 2.0  # dead time

    # Create the transfer function for the FOPDT model
    # G(s) = K / (tau * s + 1)
    numerator = [K]
    denominator = [tau, 1]
    G = ctrl.TransferFunction(numerator, denominator)

    # Apply the Pade approximation for the dead time (theta)
    num_pade, den_pade = ctrl.pade(theta, 10)  # 10th order approximation
    Pade = ctrl.TransferFunction(num_pade, den_pade)

    # Combine the transfer function with the dead time approximation
    G_deadtime = ctrl.series(Pade, G)

    # Simulate the step response
    time = np.linspace(0, 50, 500)
    t, y = ctrl.step_response(G_deadtime, T=time)

    # Plot the step response
    plt.figure()
    plt.plot(t, y)
    plt.title('Step Response of FOPDT Model with Pade Approximation')
    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.grid(True)
    plt.show()



def controls(file_paths):
    pass

def stats(new_y): 
    stdiv = np.std(new_y)
    variance = np.var(new_y)
    return stdiv,variance

    
    # cal = Calendar(main_root, selectmode='day', year=2024, month=5, day=17)
    # cal.pack(pady=20) 
    # frame_1 = customtkinter.CTkFrame(main_root,width=580, height=780, corner_radius=10, fg_color='#787474')
    # frame_1.place(relx=0.01, rely=0.01)


if __name__ == "__main__":
    global root
    root = customtkinter.CTk()
    root.withdraw()
    main_window()
    root.mainloop()



