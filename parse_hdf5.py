import h5py
import yaml
import glob
import numpy as np

from collections import defaultdict
from scipy.sparse import save_npz, csc_matrix

DATA_DIR = '/eos/user/r/rradev/nd2fd_data/larnd-sim/outputs/output_*.h5'
TARGET_DIR = '/afs/cern.ch/user/r/rradev/cvn/dune-cvn/npz_files'
consts = {
    'vdrift': 0.1648, # cm / us
    'cm2mm': 10,
    't_sampling': 0.1, # us
    'max_adc': 255,
    'min_adc': 78
}

geometry_yaml = yaml.load(open("geometry/multi_tile_layout-3.0.40.yaml"), Loader=yaml.FullLoader)
det_yaml = yaml.load(open("geometry/ndlar-module.yaml"),Loader=yaml.FullLoader)

pixel_pitch = geometry_yaml['pixel_pitch']
is_multi_tile = True
chip_channel_to_position = geometry_yaml['chip_channel_to_position']
tile_orientations = geometry_yaml['tile_orientations']
tile_positions = geometry_yaml['tile_positions']
tpc_centers = det_yaml['tpc_offsets']
tile_indeces = geometry_yaml['tile_indeces']
xs = np.array(list(chip_channel_to_position.values()))[:, 0] * pixel_pitch
ys = np.array(list(chip_channel_to_position.values()))[:, 1] * pixel_pitch
x_size = max(xs)-min(xs)+pixel_pitch
y_size = max(ys)-min(ys)+pixel_pitch

def _rotate_pixel(pixel_pos, tile_orientation):
    return pixel_pos[0]*tile_orientation[2], pixel_pos[1]*tile_orientation[1]

tile_geometry = defaultdict(int)
geometry = {}
io_group_io_channel_to_tile = {}

for tile in geometry_yaml['tile_chip_to_io']:
    tile_orientation = tile_orientations[tile]
    tile_geometry[tile] = tile_positions[tile], tile_orientations[tile]
    
    for chip in geometry_yaml['tile_chip_to_io'][tile]:
        io_group_io_channel = geometry_yaml['tile_chip_to_io'][tile][chip]
        io_group = io_group_io_channel//1000
        io_channel = io_group_io_channel % 1000
        io_group_io_channel_to_tile[(io_group, io_channel)] = tile

    for chip_channel in geometry_yaml['chip_channel_to_position']:
        chip = chip_channel // 1000
        channel = chip_channel % 1000
        io_group_io_channel = geometry_yaml['tile_chip_to_io'][tile][chip]

        io_group = io_group_io_channel // 1000
        io_channel = io_group_io_channel % 1000
        x = chip_channel_to_position[chip_channel][0] * \
            pixel_pitch - x_size / 2 
        y = chip_channel_to_position[chip_channel][1] * \
            pixel_pitch - y_size / 2
                
        x, y = _rotate_pixel((x, y), tile_orientation)
        x += tile_positions[tile][2]
        y += tile_positions[tile][1] 

        geometry[(io_group, io_channel,
                       chip, channel)] = x, y


def get_x_y_t(packets):
    xyz_adc = np.empty(shape = (len(packets)-1, 4))
    t0 = packets[0]['timestamp'].astype(int)
    assert packets[0]['packet_type'] == 7, 'Not a Trigger Packet'
    for idx, packet in enumerate(packets[1:]):
        io_group, io_channel, chip, channel = packet['io_group'], packet['io_channel'], packet['chip_id'], packet['channel_id']
        module_id = (io_group-1)//4
        io_group = io_group - (io_group-1)//4*4
        x,y = geometry[(io_group, io_channel, chip, channel)]
        tile = io_group_io_channel_to_tile[(io_group, io_channel)]
        z_offset = tpc_centers[module_id][0]*10
        x_offset = tpc_centers[module_id][2]*10
        y_offset = tpc_centers[module_id][1]*10
        x += x_offset
        y += y_offset
        xyz_adc[idx, 0] = x
        xyz_adc[idx, 1] = y

        #time corrections
        t = packet['timestamp'].astype(int) - t0
        
        drift_direction = tile_orientations[tile][0]
        z_anode = tile_positions[tile][0] #z_anode is only measurement in mm instead of cm
        vd = consts['vdrift']*consts['cm2mm']#mm /us 
        clock_period = consts['t_sampling']
        z_positions = t*drift_direction*vd*clock_period  + z_offset + z_anode
        xyz_adc[idx, 2] = z_positions 

        xyz_adc[idx, 3] = packet['dataword']
    return xyz_adc

def get_event_triggers(f):
    all_triggers = np.where(f['packets']['packet_type'] == 7) 
    triggers = all_triggers[0][1::2] #trigger ids come in pairs
    
    #add position of last packet into the triggers
    last_trig_pos = len(f['packets']) - 1
    triggers = np.append(triggers, last_trig_pos)
    return triggers

def get_simulated_packets(event_id, file):
    triggers = get_event_triggers(file)   
    idx = event_id - 1      
    low_packet_idx = triggers[idx]
    high_packet_idx = triggers[idx + 1] - 1

    if file['packets']['packet_type'][high_packet_idx - 1] == 4:
        high_packet_idx = high_packet_idx - 1 #quickfix for packet_types 4
    return file['packets'][low_packet_idx:high_packet_idx]

def get_simulated_event(event_id, file):
    packets = get_simulated_packets(event_id, file)
    event = get_x_y_t(packets)
    return event

def find_starting_x(xs):
    pixel_pitch = 3.8
    unique_xs = np.unique(sorted(xs))
    for idx, x in enumerate(unique_xs):
        top = x + 20*pixel_pitch
        sliced = unique_xs[idx:]
        counts = np.nonzero(sliced[sliced < top])[0].sum()
        if counts > 15:
            return x

def get_numpy_event(xs, ys, zs, adcs, size, pixel_pitch):
    pixel_event = np.zeros(shape=(size, size))
    mean_time = np.mean(zs)
    z_idx = (zs - mean_time + pixel_pitch/2)/pixel_pitch + size/2 
    start_x = find_starting_x(xs)
    x_idx = (xs - start_x + pixel_pitch/2)/pixel_pitch

    for z, x, adc in zip(z_idx, x_idx, adcs):
        if size > z > 0 and size > x > 0:
            pixel_event[int(x), int(z)] = adc
    return pixel_event

# Build dataset
eos_files = glob.glob(DATA_DIR)

correction = 0



for file_path in eos_files:
    eos_file = h5py.File(file_path)
    eventIDs = np.unique(eos_file['tracks']['eventID'])
    for eventID in eventIDs:
        print(eventID)
        corrected_id = eventID - correction
        try: 
            # Get event in pointcloud format
            event = get_simulated_event(event_id=corrected_id, file=eos_file)
            x, y, z, adc = event.T

            # Normalize adc counts
            adc = (adc - consts['min_adc'])/(consts['max_adc'] - consts['min_adc'])
            adc *= consts['max_adc']
            
            #Crop the event to 500, 500
            numpy_event = get_numpy_event(x, y, z, adc, size=500, pixel_pitch = 5)
            save_npz(f'{TARGET_DIR}/{corrected_id}.npz', csc_matrix(numpy_event).astype('uint8'), compressed=False)
        except:
            pass
    correction = np.max(eventIDs)


