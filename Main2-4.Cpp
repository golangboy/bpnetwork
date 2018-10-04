#include <iostream>
#include <math.h>
#include <cstring>
using namespace std;
#define TRANCOUNT 2000 //ѵ������

struct samPlae
{
	size_t inPutlen;
	size_t ouPutLen;
	double *Input;
	double *Ouput;
	double *Error;
	double Allerror;
};
class bpNetWork
{
public:
	bool save();
	bool open();
	void InitWeight(); //��ʼ��Ȩ��
	void Norlim(samPlae * samPle, size_t samPleNum);
	void setParm(double studyRat, size_t inputNum, size_t ouputNum, size_t neoNum);
	double LogSigmoid(double x);//Sigmoid���� ������е�һ��
	double Train(int n, double *input, double *output);	    //Train function
	double Result(double *InputVal, double *OutPutValue);
	double TrainOne(samPlae * inputSample, size_t inputsamNum);

private:
	double **Weight1;
	double **Weight2;
	double *Bias1;
	double *Bias2;
	double *Maxin, *Minin, *Maxout, *Minout;
	double m_Study;
	size_t m_Neonum;
	size_t m_InputNum;
	size_t m_OuputNum;
};

samPlae * MakeSample(size_t samPleNum, size_t intNum, size_t outNum)
{
	size_t i = 0;
	samPlae *samPle = new samPlae[samPleNum];
	for (i = 0; i<samPleNum; i++)
	{
		samPle[i].Error = new double[outNum];

		double *a = new double[4];
		a[0] = rand() % 10;
		a[1] = rand() % 10;

		a[2] = (a[0]+a[1])*a[0];	//�ӷ�����


		samPle[i].Input = &a[0];
		samPle[i].inPutlen = intNum;

		samPle[i].Ouput = &a[2];
		samPle[i].ouPutLen = outNum;

	}
	return samPle;
}

int main()
{
	samPlae *sam;
	bpNetWork Demo1;
	const size_t samNum = 100;


	Demo1.setParm(0.5, 2, 1, 40);
	Demo1.InitWeight();


	//Open
	Demo1.open();


	//Train
	sam = MakeSample(samNum, 1, 1);
	Demo1.Norlim(sam, samNum);
	for (size_t i = 0; i < TRANCOUNT; i++)
	{
		cout << endl << "Total:" << TRANCOUNT << "  " << "Train count(n):" << i + 1 << " " << "Error(e):" << Demo1.TrainOne(sam, samNum) << endl;
	}
	//Save
	Demo1.save();

Test:
	//Test
	while (1)
	{
		double *inVal = new double[2];
		double *ouVal = new double[1];
		cin >> inVal[0];
		cin >> inVal[1];
		Demo1.Result(inVal, ouVal);
		cout << ouVal[0] << endl;
	}
	return 0;
}

double bpNetWork::LogSigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double bpNetWork::Train(int n, double *input, double *output)
{
	size_t  j, k; //ѭ������ ����BB
	double sum = 0;
	double *hideOutput;
	double *detOutput;
	double *simHidError;
	double *detHid;
	double *finOutput;
	hideOutput = new double[m_Neonum];
	detOutput = new double[m_OuputNum];
	simHidError = new double[m_Neonum];
	detHid = new double[m_Neonum];
	finOutput = new double[m_Neonum];
	double *allError = new double[m_OuputNum];
	memset(allError, 0, m_OuputNum * sizeof(double));
	for (j = 0; j<m_Neonum; j++)
	{
		sum = 0;
		for (k = 0; k<m_InputNum + 1; k++)
		{
			if (k == m_InputNum)
				sum += (-1)*Bias1[j];
			else
				sum += input[k] * Weight1[k][j];
		}
		hideOutput[j] = LogSigmoid(sum);	//hidOutput���ز�������  ÿ���������ж�Ӧ������Ԫ���ز㣿��������
	}
	for (j = 0; j<m_OuputNum; j++)
	{
		sum = 0;
		for (k = 0; k<m_Neonum + 1; k++)
		{
			if (k == m_Neonum)
				sum += (-1)*Bias2[j];
			else
				sum += hideOutput[k] * Weight2[k][j];
		}
		finOutput[j] = LogSigmoid(sum);	//������������ֵ ��ΪԤ��ֵ
		allError[j] = output[j] - finOutput[j];
	}
	for (j = 0; j<m_OuputNum; j++)
	{
		detOutput[j] = finOutput[j] * (1 - finOutput[j])*allError[j]; //ǰ������֮��
		for (k = 0; k<m_Neonum; k++)
		{
			Weight2[k][j] += (m_Study*detOutput[j] * hideOutput[k]);
		}
		Bias2[j] += (m_Study*detOutput[j] * (-1));
	}
	for (j = 0; j<m_Neonum; j++)
	{
		simHidError[j] = 0;
		for (k = 0; k<m_OuputNum; k++)
		{
			simHidError[j] += detOutput[k] * Weight2[j][k];	//������
		}
	}
	for (j = 0; j<m_Neonum; j++)
	{
		detHid[j] = hideOutput[j] * (1 - hideOutput[j])*simHidError[j];
		for (k = 0; k<m_InputNum; k++)
		{
			Weight1[k][j] += (m_Study*detHid[j] * input[k]);
		}
		Bias1[j] += (m_Study*detHid[j] * (-1));
	}
	double All = 0;
	for (j = 0; j<m_OuputNum; j++)
		All += allError[j];

	delete[]hideOutput;
	delete[]detHid;
	delete[]detOutput;
	delete[]simHidError;
	delete[]finOutput;
	delete[]allError;

	return All;
}

double bpNetWork::Result(double *InputVal, double *OutPutValue)
{
	size_t  j, k;
	double sum = 0;
	double *hidOutput = new double[m_Neonum];
	//��һ����������
	for (j = 0; j < m_InputNum; j++)
		InputVal[j] = (InputVal[j] - Minin[j] + 1) / (Maxin[j] - Minin[j] + 1);

	//ǰ�򴫲�
	for (j = 0; j<m_Neonum; j++)
	{
		sum = 0;
		for (k = 0; k < m_InputNum; k++)
		{
			sum += InputVal[k] * Weight1[k][j];
		}
		sum += (-1)*Bias1[j];
		hidOutput[j] = LogSigmoid(sum);	//hidOutput���ز�������  ÿ���������ж�Ӧ������Ԫ���ز㣿��������
	}


	for (j = 0; j<m_OuputNum; j++)
	{
		sum = 0;
		for (k = 0; k<m_Neonum + 1; k++)
		{
			if (k == m_Neonum)
				sum += (-1)*Bias2[j];
			else
				sum += hidOutput[k] * Weight2[k][j];
		}
		OutPutValue[j] = LogSigmoid(sum)*(Maxout[j] - Minout[j] + 1) + Minout[j] - 1;	//������������ֵ ��ΪԤ��ֵ
	}


	delete[]hidOutput;
	return 0;
}

void bpNetWork::InitWeight()
{
	//InitWeight
	Bias1 = new double[m_Neonum];
	Bias2 = new double[m_OuputNum];
	Maxin = new double[m_InputNum];
	Minin = new double[m_InputNum];
	Maxout = new double[m_OuputNum];
	Minout = new double[m_OuputNum];

	Weight1 = new double*[m_InputNum];
	for (size_t i = 0; i < m_InputNum; i++)
		Weight1[i] = new double[m_Neonum];

	Weight2 = new double*[m_Neonum];
	for (size_t i = 0; i < m_Neonum; i++)
		Weight2[i] = new double[m_OuputNum];

	size_t i = 0, j = 0;
	for (i = 0; i<m_Neonum; i++)
	{
		for (j = 0; j<m_InputNum; j++)
		{
			Weight1[j][i] = (rand() / (double)RAND_MAX) * 2 - 1;
		}

	}

	for (i = 0; i<m_Neonum; i++)
	{
		Bias1[i] = (rand() / (double)RAND_MAX) * 2 - 1;
	}

	for (i = 0; i<m_OuputNum; i++)
	{
		for (j = 0; j<m_Neonum; j++)
		{
			Weight2[j][i] = (rand() / (double)RAND_MAX) * 2 - 1;
		}
	}
	for (i = 0; i<m_OuputNum; i++)
		Bias2[i] = (rand() / (double)RAND_MAX) * 2 - 1;
	//InitWeight
}

void bpNetWork::setParm(double studyRat, size_t inputNum, size_t ouputNum, size_t neoNum)
{
	if (0<studyRat && studyRat<1)
		this->m_Study = studyRat;
	else this->m_Study = 0.3;

	this->m_InputNum = inputNum;
	this->m_Neonum = neoNum;
	this->m_OuputNum = ouputNum;


}

void bpNetWork::Norlim(samPlae * samPle, size_t samPleNum)
{
	size_t i, j;
	for (i = 0; i<m_InputNum; i++)
	{
		Minin[i] = Maxin[i] = samPle[0].Input[i];
		for (j = 0; j<samPleNum; j++)
		{
			Maxin[i] = Maxin[i]>samPle[j].Input[i] ? Maxin[i] : samPle[j].Input[i];
			Minin[i] = Minin[i]<samPle[j].Input[i] ? Minin[i] : samPle[j].Input[i];
		}
	}
	for (i = 0; i<m_OuputNum; i++) {
		Minout[i] = Maxout[i] = samPle[0].Ouput[i];
		for (j = 0; j<samPleNum; j++)
		{
			Maxout[i] = Maxout[i]>samPle[j].Ouput[i] ? Maxout[i] : samPle[j].Ouput[i];
			Minout[i] = Minout[i]<samPle[j].Ouput[i] ? Minout[i] : samPle[j].Ouput[i];
		}
	}

	for (i = 0; i < m_InputNum; i++)
		for (j = 0; j < samPleNum; j++)
			samPle[j].Input[i] = (samPle[j].Input[i] - Minin[i] + 1) / (Maxin[i] - Minin[i] + 1);

	for (i = 0; i < m_OuputNum; i++)
		for (j = 0; j < samPleNum; j++)
			samPle[j].Ouput[i] = (samPle[j].Ouput[i] - Minout[i] + 1) / (Maxout[i] - Minout[i] + 1);
}

double bpNetWork::TrainOne(samPlae * inputSample, size_t inputsamNum)
{
	size_t i, j;
	double MSE = 0;
	for (i = 0; i<inputsamNum; i++)
	{
		inputSample[i].Allerror = Train(i, inputSample[i].Input, inputSample[i].Ouput);
	}
	MSE = 0;
	for (i = 0; i<m_OuputNum; i++)
	{
		for (j = 0; j<inputsamNum; j++)
		{
			MSE += (inputSample[j].Allerror  * inputSample[j].Allerror) / 2;
		}
	}
	return MSE;
}

bool bpNetWork::save()
{
	FILE *hFile;
	hFile = fopen("save.txt", "wb");
	if (nullptr == hFile)
	{
		return false;
	}
	size_t nCount = 0;
	size_t allSize = m_InputNum * m_Neonum + m_Neonum * m_OuputNum + m_Neonum + m_OuputNum + m_InputNum * 2 + m_OuputNum * 2;
	double *saveData = new double[allSize];

	size_t i, j, k;
	for (j = 0; j < m_Neonum; j++)
	{
		for (k = 0; k < m_InputNum; k++)
		{
			saveData[nCount++] = Weight1[k][j];
		}
	}

	for (j = 0; j < m_OuputNum; j++)
	{
		for (k = 0; k < m_Neonum; k++)
		{
			saveData[nCount++] = Weight2[k][j];
		}
	}

	for (i = 0; i < m_Neonum; i++)
	{
		saveData[nCount] = Bias1[i];
		nCount++;
	}

	for (i = 0; i < m_OuputNum; i++)
		saveData[nCount++] = Bias2[i];

	//IN
	for (i = 0; i < m_InputNum; i++)
		saveData[nCount++] = Maxin[i];
	//IN
	for (i = 0; i < m_InputNum; i++)
		saveData[nCount++] = Minin[i];
	//OUT
	for (i = 0; i < m_OuputNum; i++)
		saveData[nCount++] = Maxout[i];
	//OUT
	for (i = 0; i < m_OuputNum; i++)
		saveData[nCount++] = Minout[i];

	fwrite(saveData, allSize * sizeof(double), 1, hFile);
	fclose(hFile);
	delete[]saveData;
	return true;
}

bool bpNetWork::open()
{
	FILE *hFile;
	hFile = fopen("save.txt", "rb");
	if (nullptr == hFile)
	{
		return false;
	}

	size_t nCount = 0;
	size_t allSize = m_InputNum * m_Neonum + m_Neonum * m_OuputNum + m_Neonum + m_OuputNum + m_InputNum * 2 + m_OuputNum * 2;
	double *saveData = new double[allSize];
	memset(saveData, 0, allSize * sizeof(double));
	fread(saveData, allSize * sizeof(double), 1, hFile);

	size_t j, k, i;
	for (j = 0; j < m_Neonum; j++)
	{
		for (k = 0; k < m_InputNum; k++)
		{
			Weight1[k][j] = saveData[nCount++];
		}
	}

	for (j = 0; j < m_OuputNum; j++)
	{
		for (k = 0; k < m_Neonum; k++)
		{
			Weight2[k][j] = saveData[nCount++];
		}
	}

	for (i = 0; i < m_Neonum; i++)
		Bias1[i] = saveData[nCount++];

	for (i = 0; i < m_OuputNum; i++)
		Bias2[i] = saveData[nCount++];

	//IN
	for (i = 0; i < m_InputNum; i++)
		Maxin[i] = saveData[nCount++];
	//IN
	for (i = 0; i < m_InputNum; i++)
		Minin[i] = saveData[nCount++];
	//OUT
	for (i = 0; i < m_OuputNum; i++)
		Maxout[i] = saveData[nCount++];
	//OUT
	for (i = 0; i < m_OuputNum; i++)
		Minout[i] = saveData[nCount++];


	fclose(hFile);

	return true;

}
