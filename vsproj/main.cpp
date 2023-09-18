
#include "dajmodel.h"

using namespace dajnn;

int main(int argc, const char** argv) {
	Model* model = new Model("../../pymaster/duel/export/koni_p2_d5.daj");
	delete model;
	return 0;
}
