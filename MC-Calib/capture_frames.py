import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 1. Find connected RealSense devices
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    
    if len(serials) < 2:
        print(f"Error: Found {len(serials)} camera(s). Please connect at least 2 cameras.")
        return

    print(f"Found cameras with serials: {serials[0]} and {serials[1]}")

    pipelines = []
    aligns = []

    # 2. Configure and start both cameras
    for serial in serials[:2]: # Limit to first 2 cameras to match your calibration
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        
        # Set resolution to match your calibration intrinsics
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        pipeline.start(config)
        pipelines.append(pipeline)
        
        # Create an align object to map depth coordinates to color coordinates
        align_to = rs.stream.color
        aligns.append(rs.align(align_to))

    print("Streaming... Press 's' to save frames, or 'q' to quit.")

    try:
        while True:
            frameset = []
            valid_frames = True
            
            # Fetch frames from both cameras
            for i, pipeline in enumerate(pipelines):
                frames = pipeline.wait_for_frames()
                aligned_frames = aligns[i].process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    valid_frames = False
                    break
                    
                frameset.append((color_frame, depth_frame))

            if not valid_frames:
                continue

            # Convert images to numpy arrays for OpenCV
            color_images = [np.asanyarray(f[0].get_data()) for f in frameset]
            depth_images = [np.asanyarray(f[1].get_data()) for f in frameset]

            # Apply colormap to depth images purely for visualization (we save the raw ones later)
            depth_colormaps = [cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=0.03), cv2.COLORMAP_JET) for d in depth_images]

            # Stack images horizontally for a nice preview window
            cam0_view = np.vstack((color_images[0], depth_colormaps[0]))
            cam1_view = np.vstack((color_images[1], depth_colormaps[1]))
            preview = np.hstack((cam0_view, cam1_view))
            
            # Resize preview so it fits on standard screens
            preview_resized = cv2.resize(preview, (1280, 720))
            cv2.imshow('RealSense Multi-Camera Preview (Press S to save)', preview_resized)

            key = cv2.waitKey(1)
            
            # If 's' is pressed, save the frames and exit
            if key & 0xFF == ord('s'):
                print("Saving images...")
                # Save Cam 0
                cv2.imwrite("cam0_color.png", color_images[0])
                cv2.imwrite("cam0_depth.png", depth_images[0]) # Saves as 16-bit natively
                # Save Cam 1
                cv2.imwrite("cam1_color.png", color_images[1])
                cv2.imwrite("cam1_depth.png", depth_images[1]) # Saves as 16-bit natively
                
                print("Images saved successfully! You can now run reprojection.py")
                break
                
            # If 'q' or Esc is pressed, quit without saving
            elif key & 0xFF == ord('q') or key == 27:
                print("Exiting...")
                break

    finally:
        # Stop all pipelines
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()