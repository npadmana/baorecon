#include <iostream>
#include <string>
#include "ticpp.h"

using namespace std;

int main() {
  double val=0.0;
  ticpp::Document doc("example.xml");
  doc.LoadFile();
  ticpp::Iterator< ticpp::Element > nodes;
  for (nodes=nodes.begin(doc.FirstChildElement()); nodes !=nodes.end(); ++nodes) {
    nodes->GetAttribute("val", &val);
    cout << nodes->Value() << " " <<  val << endl;
  }

  cout << doc.FirstChildElement()->FirstChildElement("N")->GetAttribute("val") << endl;

  return 0;
}
