#include "scene_file_proc.h"

inline void readHelioParamter(stringstream &scene_stream, Heliostat *heliostat) {
	std::string line_buf;
	getline(scene_stream, line_buf);
	float3 size;
	std::stringstream line_stream_buf;
	line_stream_buf << line_buf;
	line_stream_buf >> size.x >> size.y >> size.z;
	heliostat->size_ = size;
};




SceneFileProc::SceneFileProc() {
	string_value_read_map["pos"] = StringValue::pos;
	string_value_read_map["size"] = StringValue::size;
	string_value_read_map["norm"] = StringValue::norm;
	string_value_read_map["face"] = StringValue::face;
	string_value_read_map["end"] = StringValue::end;
	string_value_read_map["gap"] = StringValue::gap;
	string_value_read_map["matrix"] = StringValue::matrix;
	string_value_read_map["helio"] = StringValue::helio;
	string_value_read_map["inter"] = StringValue::inter;
	string_value_read_map["n"] = StringValue::n;
	string_value_read_map["type"] = StringValue::type;
}

StringValue SceneFileProc::Str2Value(string str) {
	if (string_value_read_map.count(str)) {
		return string_value_read_map[str];
	}
	else {
		std::cerr << str << " is not define in the string_value_read_map" << std::endl;
		return StringValue::illegal;
	}
}


bool SceneFileProc::SceneFileRead(SolarScene *solarscene, std::string filepath) {
	Receiver *receiver;
	RectGrid *grid0;
	Heliostat *heliostat;
	//to save the global settings
	int helio_type_buf;
	bool gap_set = false;
	bool matrix_set = false;
	float2 gap_buf;
	int2 matrix_buf;
	// prama to ensure the memery
	int helio_input_num = 0;
	int helio_input_total = 0;
	int grid_start_helio_pos = 0;

	InputMode inputMode = InputMode::none;
	solarScene_ = solarscene;
	std::string str_line;
	std::ifstream scene_file;
	// ensure ifstream objects can throw exceptions:
	scene_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try {
		//openfile
		scene_file.open(filepath);
		stringstream scene_stream;
		// read file's buffer contents into streams
		scene_stream << scene_file.rdbuf();
		//close the file handlers
		scene_file.close();
		while (getline(scene_stream, str_line)) {
			if (str_line[0] == '#' || str_line == "") { continue; }
#ifdef _DEBUG
	//		std::cout << str_line << std::endl;
#endif
			std::stringstream line_stream;
			line_stream << str_line;  
			//get the line flag
			std::string line_head;
			line_stream >> line_head;

			//chose the mode to read
			if (line_head == "ground") {  
				inputMode = InputMode::ground;
				float ground_length, ground_width;
				line_stream >> ground_length >> ground_width;
				solarScene_->ground_length_ = ground_length;
				solarScene_->ground_width_ = ground_width;
				continue;
			}
			else if (line_head == "Recv") {  // init the receiver
				inputMode = InputMode::receiver;
				int receive_type;
				line_stream >> receive_type;
				switch (receive_type) 
				{
					case 0: receiver = new RectangleReceiver();
						break;
					case 1: receiver = new CylinderReceiver();
						break;
					case 2: receiver = new CircularTruncatedConeReceiver();
						break;
					default:
						std::cerr << "Receiver type not define!" << std::endl;
						break;
				}
				receiver->type_ = receive_type;
				continue;

			}
			else if (line_head == "Grid"){
				//vertify the last grid
				if (helio_input_num != helio_input_total) {
					std::cerr << "helistat number is wrong" << std::endl;
				}
				helio_input_num = 0;
				helio_input_total = 0;

				inputMode = InputMode::grid;
				int grid_type;
				line_stream >> grid_type;
				switch (grid_type)
				{
				case 0: grid0 = new RectGrid();
					break;
				case 1:
					break;
				default:
					std::cerr << "Grid type not define!" << std::endl;
					break;
				}
				grid0->type_ = grid_type;
				helio_input_num = 0;
				grid0->start_helio_pos_ = grid_start_helio_pos;
				//reset the gap 
				gap_set = false;
				matrix_set = false;
				continue;
			}
			/*************  switch section  ************************/
			switch (inputMode)
			{
				case InputMode::none:
#ifdef _DEBUG
					std::cout << "none mode: ignore this line" << std::endl;
#endif
					break;
				case InputMode::ground:
					int grid_num;
					if (line_head == "ngrid") {
						line_stream >> grid_num;
						solarScene_->grid_num_ = grid_num;
					}
					break;
				case InputMode::receiver:
					switch (string_value_read_map[line_head]) {
						case StringValue::pos:
							float3 pos;
							line_stream >> pos.x >> pos.y >> pos.z;
							receiver->pos_ = pos;
							break;
						case StringValue::size:
							float3 size;
							line_stream >> size.x >> size.y >> size.z;
							receiver->size_ = size;
							break;
						case StringValue::norm:
							float3 norm;
							line_stream >> norm.x >> norm.y >> norm.z;
							receiver->normal_ = norm;
							break;
						case StringValue::face:
							int face_num;
							line_stream >> face_num;
							receiver->face_num_ = face_num;
							break;
						case StringValue::end: //push the receiver
							solarScene_->receivers.push_back(receiver);  
							receiver = nullptr;
							break;
						default:
							break;
					}

					break;
				case InputMode::grid:
					if (grid0->type_ == 0) {
						switch (string_value_read_map[line_head]) {
						case StringValue::pos:
							float3 pos;
							line_stream >> pos.x >> pos.y >> pos.z;
							grid0->pos_ = pos;
							break;
						case StringValue::size:
							float3 size;
							line_stream >> size.x >> size.y >> size.z;
							grid0->size_ = size;
							break;
						case StringValue::inter:
							float3 inter;
							line_stream >> inter.x >> inter.y >> inter.z;
							grid0->interval_ = inter;
							break;
						case StringValue::n:   //update the helio size
							int n;
							line_stream >> n;
							grid0->num_helios_ = n;
							helio_input_total = n;
							grid_start_helio_pos += n;
							break;
						case StringValue::type:
							int helio_type;
							line_stream >>helio_type;
							grid0->helio_type_ = helio_type;
							break;
						case StringValue::end:
							solarScene_->grid0s.push_back(grid0);
							// change the mode
							inputMode = InputMode::heliostat;
							helio_type_buf = grid0->helio_type_;
							//delete grid0;
							grid0 = nullptr;
							break;
						default:
							break;
						}
					}
					else {
						std::cerr << "this type grid is not support";
					}
					break;
				case InputMode::heliostat:
					switch (string_value_read_map[line_head]) {
						case StringValue::gap:
							float2 gap;
							line_stream >> gap.x >> gap.y;
							gap_buf = gap;
							gap_set = true;
							break;
						case StringValue::matrix:
							int2 matrix;
							line_stream >> matrix.x >> matrix.y;
							matrix_buf = matrix;
							matrix_set = true;
							break;
						case StringValue::helio:
							if (!gap_set || !matrix_set) {
								std::cerr << "did not set the gap and matrix"<< std::endl;
								break;
							}
							//ensure the member
							if (helio_input_num >= helio_input_total) {
								std::cerr << "too many helistat" << std::endl;
							}
							if (helio_type_buf == 0) {
								heliostat = new RectangleHelio;
								float3 pos;
								line_stream >> pos.x >> pos.y >> pos.z;
								heliostat->pos_ = pos;
								heliostat->gap_ = gap_buf;//gap
								heliostat->row_col_ = matrix_buf;//matrix
								if (helio_type_buf == 0) {
									readHelioParamter(scene_stream, heliostat);
								}
								solarScene_->heliostats.push_back(heliostat);
								heliostat = nullptr; //make sure heliostat is null
								++helio_input_num;
							}
							break;
						default:
							break;
					}
					break;
				default:
					std::cout << "illeagle parameter" << std::endl;
					break;
			}
		}

	}
	catch (std::ifstream::failure e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
		return false;
	}
	return true;
}

bool SceneFileProc::SceneFileRead(SolarScene *solarscene, float a,float b,float *mirrorNormal,float X,float Y,float Z) {
	float centerX[9] = { -sqrt(2.0) / 3 - 0.1 ,0 ,sqrt(2.0) / 3 + 0.1, -sqrt(2.0) / 3 - 0.1,0 ,sqrt(2.0) / 3 + 0.1, -sqrt(2.0) / 3 - 0.1, 0 ,sqrt(2.0) / 3 + 0.1 };
	float centerZ[9] = { -sqrt(2.0) / 3 - 0.1, -sqrt(2.0) / 3 - 0.1 ,-sqrt(2.0) / 3 - 0.1, 0, 0 ,0 ,sqrt(2.0) / 3 + 0.1 ,sqrt(2.0) / 3 + 0.1, sqrt(2.0) / 3 + 0.1 };
 
	float centerY[9] = { 0 };
	float centerT[3], center[3];
	float normalT[3],normal[3];
	float temp;
	float mirrorToGround[3][3];
	float3 normals[9];
	for (int i = 0; i < 9; i++) { 
		centerY[i] = centerX[i] * centerX[i] / a / a+ centerZ[i] * centerZ[i] / b / b;
	}
	
	temp = sqrt(mirrorNormal[0] * mirrorNormal[0] + mirrorNormal[1] * mirrorNormal[1]);
	if (mirrorNormal[2] > 0) {
		mirrorToGround[0][0] = mirrorNormal[2] / temp;
		mirrorToGround[2][0] = -mirrorNormal[0] / temp;
		mirrorToGround[1][0] = 0;

	}
	else if (mirrorNormal[2] < 0) {
		mirrorToGround[0][0] = -mirrorNormal[2] / temp;
		mirrorToGround[2][0] = mirrorNormal[0] / temp;
		mirrorToGround[1][0] = 0;

	}
	else {
		mirrorToGround[0][0] = 0;
		mirrorToGround[1][0] = 0;
		mirrorToGround[2][0] = 1;
	} 
	for (int i = 0; i < 3; i++) {
		mirrorToGround[i][1] = mirrorNormal[i];

	}
	mirrorToGround[0][2] = mirrorToGround[1][0] * mirrorToGround[2][1] - mirrorToGround[2][0] * mirrorToGround[1][1];
	mirrorToGround[1][2] = mirrorToGround[2][0] * mirrorToGround[0][1] - mirrorToGround[0][0] * mirrorToGround[2][1];
	mirrorToGround[2][2] = mirrorToGround[0][0] * mirrorToGround[1][1] - mirrorToGround[1][0] * mirrorToGround[0][1];

	for (int i = 0; i < 9; i++) {
		center[0] = centerX[i];
		center[1] = centerY[i];
		center[2] = centerZ[i];
		normal[0]=-center[0]*2/a/a;
		normal[1]=1;
		normal[2]=-center[2]*2/b/b;
		matrixMulti(mirrorToGround, center, centerT);
		centerX[i] = centerT[0]+X;
		centerY[i] = centerT[1]+Y;
		centerZ[i] = centerT[2]+Z;
		matrixMulti(mirrorToGround,normal,normalT);
		normals[i]=make_float3(normalT[0],normalT[1],normalT[2]);
	}
	Receiver *receiver;
	RectGrid *grid0;
	Heliostat *heliostat;
	//to save the global settings
	int helio_type_buf;
	bool gap_set = false;
	bool matrix_set = false;
	float2 gap_buf;
	int2 matrix_buf;
	// prama to ensure the memery
	int helio_input_num = 0;
	int helio_input_total = 0;
	int grid_start_helio_pos = 0;

	InputMode inputMode = InputMode::none;
	solarScene_ = solarscene;
		int i = 1;
		while (i<5) {
			//chose the mode to read
			if (i == 1) {
				inputMode = InputMode::ground;
				float ground_length=1024.0, ground_width=1024.0;
				solarScene_->ground_length_ = ground_length;
				solarScene_->ground_width_ = ground_width;
			}
			else if (i == 2) {  // init the receiver
				inputMode = InputMode::receiver;
				int receive_type=0;
				switch (receive_type)
				{
				case 0: receiver = new RectangleReceiver();
					break;
				case 1: receiver = new CylinderReceiver();
					break;
				case 2: receiver = new CircularTruncatedConeReceiver();
					break;
				default:
					std::cerr << "Receiver type not define!" << std::endl;
					break;
				}
				receiver->type_ = receive_type;

			}
			else if (i == 3) {
				//vertify the last grid
				if (helio_input_num != helio_input_total) {
					std::cerr << "helistat number is wrong" << std::endl;
				}
				helio_input_num = 0;
				helio_input_total = 0;

				inputMode = InputMode::grid;
				int grid_type;
				grid_type = 0;
				switch (grid_type)
				{
				case 0: grid0 = new RectGrid();
					break;
				case 1:
					break;
				default:
					std::cerr << "Grid type not define!" << std::endl;
					break;
				}
				grid0->type_ = grid_type;
				helio_input_num = 0;
				grid0->start_helio_pos_ = grid_start_helio_pos;
				//reset the gap 
				gap_set = false;
				matrix_set = false;
			}
			/*************  switch section  ************************/
			switch (inputMode)
			{
			case InputMode::none:
				break;
			case InputMode::ground:
				solarScene_->grid_num_ =1;				
				break;

			case InputMode::receiver:
				float3 pos;
				pos.x = 0.0;
				pos.y = 13.0;
				pos.z = -1.50;
				receiver->pos_ = pos;

				float3 size;
				size.x =1.4;
				size.y = 1.4;
				size.z = 3.0;
				receiver->size_ = size;

				float3 norm;
				norm.x = 0.0;
				norm.y = 0.0;
				norm.z = 1.0;
				receiver->normal_ = norm;
									
				receiver->face_num_ = 0;
				solarScene_->receivers.push_back(receiver);
				receiver = nullptr;
				break;

			case InputMode::grid:
				if (grid0->type_ == 0) {
						float3 pos;
						pos.x = -5.5;
						pos.y = -5.5;
						pos.z = 94.5;
						grid0->pos_ = pos;
					
						float3 size;
						size.x = 110.0;
						size.y = 110.0;
						size.z = 220.0;
						grid0->size_ = size;
						

						float3 inter;
						inter.x = 110.0;
						inter.y = 110.0;
						inter.z = 110.0;
						grid0->interval_ = inter;
										
						int n=9;
						grid0->num_helios_ = n;
						helio_input_total = n;
						grid_start_helio_pos += n;

						int helio_type=0;
						grid0->helio_type_ = helio_type;
						
						solarScene_->grid0s.push_back(grid0);
						// change the mode
						inputMode = InputMode::heliostat;
						helio_type_buf = grid0->helio_type_;
						//delete grid0;
						grid0 = nullptr;				
				}
				else {
					std::cerr << "this type grid is not support";
				}
				break;
			case InputMode::heliostat:
				while (helio_input_num<helio_input_total){
					float2 gap;
					gap.x = 0.0;
					gap.y = 0.0;
					gap_buf = gap;
					gap_set = true;
					int2 matrix;
					matrix.x = 1;
					matrix.y = 1;
					matrix_buf = matrix;
					matrix_set = true;
					if (!gap_set || !matrix_set) {
						std::cerr << "did not set the gap and matrix" << std::endl;
						break;
					}
					//ensure the member
					if (helio_input_num >= helio_input_total) {
						std::cerr << "too many helistat" << std::endl;
					}
					if (helio_type_buf == 0) {
						heliostat = new RectangleHelio;
						float3 pos,size;
						pos.x = centerX[helio_input_num];
						pos.y = centerY[helio_input_num];
						pos.z = centerZ[helio_input_num];
						heliostat->pos_ = pos;
						heliostat->gap_ = gap_buf;//gap
						heliostat->row_col_ = matrix_buf;//matrix
						size.x = 1.4142136/3;
						size.y = 0.1;
						size.z = 1.4142136/3;
						heliostat->size_ = size;
						solarScene_->heliostats.push_back(heliostat);
						heliostat = nullptr; //make sure heliostat is null
						++helio_input_num;
					}
				}
				break;

			default:
				std::cout << "illeagle parameter" << std::endl;
				break;
			}
			i++;
		}
		solarScene_ ->InitContent(normals);
	return true;
}

void SceneFileProc::matrixMulti(float A[][3], float *b, float *c) {
	for (int i = 0; i < 3; i++) {
		c[i] = 0;
		for (int j = 0; j < 3; j++) {
			c[i] += A[i][j] * b[j];
		}
	}
}