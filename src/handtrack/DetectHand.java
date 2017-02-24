package handtrack;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class DetectHand {
	
	private static final int FRAME_HEIGHT = 768;
	private static final int FRAME_WIDTH = 1024;
	
	private static boolean close = false;
	
	private JFrame frame = new JFrame("Hand track");
	private JLabel lab = new JLabel();
	
	public DetectHand(){}

	public static void main(String[] args)
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		DetectHand d = new DetectHand();
		VideoCapture webcam = new VideoCapture(0);
		Mat frame = new Mat();
		
		d.setFrame(webcam);
		frame = d.getFrame(webcam);
		frame = d.findColor(frame);
		
		d.printFrame(frame);
		
		
		while(true)
		{
			frame = d.getFrame(webcam);
			frame = d.findColor(frame);
			d.printFrame(frame);
		}
		
		
	}
	
	/**
	 * Hand tracking is done by color. We want to take an average
	 * color of the whole hand. This way we can recognize the hand
	 * and keep out unwanted blur.
	 */
	public Mat findColor(Mat frame)
	{
		int startX = 400;
		int startY = 50;
		int lengthX = 330;
		int lengthY = 550;
		
		Rect rect = new Rect(startX, startY, lengthX, lengthY);
		Mat area = frame.submat(rect);
		Mat skin = skinDetection(frame);
		return skin;
	}
	
	public Mat skinDetection(Mat orig) {
		Mat mask = new Mat();
		Mat result = new Mat();
		Core.inRange(orig, new Scalar(0, 0, 0), new Scalar(30, 30, 30), result);
		Imgproc.cvtColor(orig, mask, Imgproc.COLOR_BGR2HSV);
		for (int i = 0; i < mask.size().height; i++) {
			for (int j = 0; j < mask.size().width; j++) {
				if (mask.get(i, j)[0] < 19
						|| mask.get(i, j)[0] > 150 && mask.get(i, j)[1] > 25 && mask.get(i, j)[1] < 220) {

					result.put(i, j, 255, 255, 255);

				} else {
					result.put(i, j, 0, 0, 0);
				}
			}

		}
		return process(result);
	}
	
	public Mat process(Mat orig)
	{
		Mat frame = orig;
		//Imgproc.blur(frame, frame, new Size(7,7));
		Imgproc.medianBlur(frame, frame, 3);
		
		Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));
		Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));

		Imgproc.erode(frame, frame, erodeElement);
		
		Imgproc.dilate(frame, frame, dilateElement);
		
		Imgproc.erode(frame, frame, erodeElement);

		Imgproc.dilate(frame, frame, dilateElement);
		
		findContours(frame);
		return frame;
	}
	
	public void addBox(Mat frame)
	{
		//draw box to place hand in
		//top line
		Imgproc.line(frame, new Point(400, 50), new Point(730, 50), new Scalar(255, 0, 0), 2);
		//bottom line
		Imgproc.line(frame, new Point(400, 600), new Point(730, 600), new Scalar(255, 0, 0), 2);
		//left line
		Imgproc.line(frame, new Point(400, 50), new Point(400, 600), new Scalar(255, 0, 0), 2);
		//right line
		Imgproc.line(frame, new Point(730, 50), new Point(730, 600), new Scalar(255, 0, 0), 2);
	}
	
	public void findContours(Mat frame)
	{
		List<MatOfPoint> contours = new ArrayList<>();
		Mat hierarchy = new Mat();

		// find contours
		Imgproc.findContours(frame, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

		// if any contour exist...
		if (hierarchy.size().height > 0 && hierarchy.size().width > 0)
		{
		        // for each contour, display it in blue
		        for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0])
		        {
		                Imgproc.drawContours(frame, contours, idx, new Scalar(250, 0, 0));
		        }
		}
	}
	
	public void printFrame(Mat frame)
	{
		MatOfByte cc = new MatOfByte();
		Imgcodecs.imencode(".JPG", frame, cc);
		byte[] chupa = cc.toArray();
		InputStream ss = new ByteArrayInputStream(chupa);
		try {
			BufferedImage aa = ImageIO.read(ss);
			lab.setIcon(new ImageIcon(aa));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void setFrame(final VideoCapture webcam) {
		frame.setSize(FRAME_WIDTH, FRAME_HEIGHT);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		frame.getContentPane().add(lab);
		frame.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				System.out.println("Closed");
				close = true;
				webcam.release();
				e.getWindow().dispose();
			}
		});
		
	}
	
	public Mat getFrame(VideoCapture cap)
	{
		cap.set(Videoio.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT); 
		cap.set(Videoio.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
		
		Mat frame = new Mat();
		cap.read(frame);
		addBox(frame);
		return frame;
	}
	
	

}
