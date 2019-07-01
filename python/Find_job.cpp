//
//  main.cpp
//  Find_Jobs
//
//  Created by MacBook on 2019/6/26.
//  Copyright © 2019 MacBook. All rights reserved.
//

#include <iostream>
#include <cstdio>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <algorithm>
#include <cstring>

using namespace std;

//学校
class University {
	string name="";
	int Agree_to_work=0;
	float GPA;
	//AK==1 表示同意学生工作
	
public:
	University(){}
	
	University(string name){
		this->name=name;
	}
	
	string get_name(){
		return name;
	}
	
	int get_AK(){
		return Agree_to_work;
	}
	
	void set_AK(int x){
		this->Agree_to_work=x;
		
	}
	
	void if_AK(float GPA){
		this->GPA=GPA;
		if (this->GPA>=2.5) {
			set_AK(1);
		}
	}
	
};

//求职者类
class JobSeeker {
	//是否接受信息,1 表示接收并应聘,0 表示忽略信息
	//message表示收到的求职信息
	int is_recive_work=0;
	string message;
	//求职基本信息
	string name="";
	int age;//年龄应该d大于18岁才能合法工作
	int sex;//sex==1表示男,sex==-1 表示女,sex==0表示性别未知
	JobCenter jobcenter;
	
public:
	JobSeeker();
	void set_name(string name){
		this->name=name;
		
	}
	void set_sex(int x){
		this->sex=x;
	}
	
	void set_age(int x){
		this->age=x;
	}
	
	string get_message(){
		return message;
	}
	
	void set_message(string message){
		this->message=message;
	}
	
	void set_JobCenter(JobCenter job){
		this->jobcenter=job;
	}
	string feedback_message(){
		return message;
	}
	
	int feedback_is_work(){
		return is_recive_work;
	}
};



//学生
class Student :public JobSeeker {
	University university;
	float GPA;
public:
	Student(){
		
	}

	Student(string name,int age,float GPA,University university,JobCenter jobcenter){
		set_name(name);
		set_age(age);
		set_GPA(GPA);
		set_University(university);
		set_JobCenter(jobcenter);
	}
	void set_GPA(float x){
		this->GPA=x;
	}
	
	void set_University(University university){
		this->university=university;
	}

};



//有经验的求职者
class Experienced:public JobSeeker {
	Experienced(){}
	Experienced(string name,int age,JobCenter jobcenter){
		set_name(name);
		set_age(age);
		set_JobCenter(jobcenter);
	}
	
public:
	
};


//求职中心
class JobCenter {
private:
	vector<JobSeeker> has_work;
	vector<JobSeeker> Free_work;
	string message;
	int len_free=0;
	int len_has_work=0;
	int numJobSeeker;
public:
	JobCenter();
	void publishMessage(){
		
	}
	void publishMessage(string message){
		for (int i=0; i<len_free; i++) {
			Free_work[i].set_message(message);
		}
	}
	void notifyJobSeekers(){
		for (int i=0; i<len_has_work; i++) {
			cout<<has_work[i].get_message()<<endl;
		}
	}
	
	void add_jobSeeker(JobSeeker jobseeker){
		Free_work.push_back(JobSeeker);
		len_free++;
	}
	
};



int main(int argc, const char * argv[]) {
	// insert code here...
	JobCenter jobCenter;
	University ouc("中国海洋大学");
	
	Student *zhang =new Student("小张",22,3.3,ouc,jobCenter);
	cout << "Hello, World!\n";
	return 0;
}
