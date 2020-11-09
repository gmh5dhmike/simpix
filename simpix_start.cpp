// simple example of using ROOT libraries in a C++ program with graphics
// and use of TASImage class

#include "TROOT.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TASImage.h"
#include "TApplication.h"
#include "TSystem.h"


#include "assert.h"

#include <iostream>
#include <stdio.h>
using namespace std;

int main(int argc, char **argv){

  if (argc<3) {
    cout << "Usage: simapix_start image1 image2 <output=out.png>" << endl;
    return 0; 
  }
  TString fsrc=argv[1];
  TString ftgt=argv[2];
  TString fout;
  argc>3 ? fout = argv[3] : fout="out.png";
  cout << "Reading images: source= " << fsrc << " target= " << ftgt << endl;
  cout << "Output= " << fout << endl;

  TApplication theApp("App", &argc, argv);

  // create image objects
  TASImage *src = new TASImage(fsrc.Data());
  TASImage *tgt = new TASImage(ftgt.Data());
  TASImage *out = new TASImage(*src); // start with copy of source

  // Test image geometry, exit if they are not the same dimensions
  assert ( src->GetWidth() == tgt->GetWidth() && src->GetHeight() == tgt->GetHeight() );
  cout << "Pixel Geometry: " << src->GetWidth() << " x " << src->GetHeight() << endl;
  Long_t numPix=src->GetWidth()*src->GetHeight();

  // *** The work happens here
  // access the pixels for the output image 
  // each pixel is a 32-bit word, 1 byte each for (alpha,red,green,blue)
  // don't touch alpha (bits 31:28)
  UInt_t *outPix = out->GetArgbArray();  

  // examples of pixel manipulations 
  for (int i=0;i< numPix; i++){
    //  outPix[i]&=0xff00ffff;  // turn off red
    outPix[i]&=0xffff00ff;  // turn off green
    //  outPix[i]&=0xffffff00;  // turn off blue
    //  cout << hex << outPix[i]<<endl;  // print pixel values in hex
  }
  // flip the image
  for (int i=0;i< numPix/2; i++){
    unsigned pxl=outPix[i];
    outPix[i]=outPix[numPix-i-1];
    outPix[numPix-i-1]=pxl;
  }

  // *************************


  // print the results
  TCanvas *c1 = new TCanvas("c1", "images", 640, 480);
  c1->Divide(2,2);

  c1->cd(1);
  c1->Draw();
  src->Draw("X");
  c1->cd(2);
  tgt->Draw("X");
  c1->cd(3);
  out->Draw("X");
  c1->Print("collage.png");
  
  // save the new image
  out->WriteImage(fout.Data());

  // coment out the lines for running in batch mode
  cout << "Press ^c to exit" << endl;
  theApp.SetIdleTimer(30,".q");  // set up a failsafe timer to end the program  
  theApp.Run();

}
