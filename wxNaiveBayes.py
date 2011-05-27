#!/usr/bin/env python
# -*- coding: utf-8 -*-
# generated by wxGlade 0.6.3 on Sat May 28 02:02:52 2011

import wx
import naivebayes

# begin wxGlade: extracode
# end wxGlade

nb = naivebayes.NaiveBayes()

class wxNaiveBayes(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: wxNaiveBayes.__init__
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.rb1 = wx.RadioButton(self, -1, u"訓練")
        self.rb2 = wx.RadioButton(self, -1, u"分類")
        self.Display = wx.StaticText(self, -1, "label_1")
        self.btnOpen = wx.Button(self, wx.ID_OPEN, "")
        self.btnApply = wx.Button(self, wx.ID_APPLY, "")
        self.btnClose = wx.Button(self, wx.ID_CLOSE, "")

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_BUTTON, self.onOpen, self.btnOpen)
        self.Bind(wx.EVT_BUTTON, self.onApply, self.btnApply)
        self.Bind(wx.EVT_BUTTON, self.onClose, self.btnClose)
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: wxNaiveBayes.__set_properties
        self.SetTitle(u"ナイーブベイズ分類器")
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: wxNaiveBayes.__do_layout
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_2 = wx.BoxSizer(wx.VERTICAL)
        sizer_3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_2.Add(self.rb1, 0, 0, 0)
        sizer_2.Add(self.rb2, 0, 0, 0)
        sizer_2.Add(self.Display, 6, 0, 0)
        sizer_3.Add(self.btnOpen, 0, 0, 0)
        sizer_3.Add(self.btnApply, 0, 0, 0)
        sizer_3.Add(self.btnClose, 0, 0, 0)
        sizer_2.Add(sizer_3, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_2, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()
        # end wxGlade

    def onOpen(self, event): # wxGlade: wxNaiveBayes.<event_handler>
        # チェックボックスの情報を感知
        if self.rb1.GetValue():
            # 訓練にチェックが入ってた場合
            trainDlg = trainDialog(None, -1, "")
            result  = trainDlg.ShowModal()
            
            if result == wx.ID_OK:
                doc = trainDlg.text_ctrl_1.GetValue().encode('utf-8')
                print doc               # デバッグ用プリント
                cat = trainDlg.text_ctrl_2.GetValue().encode('utf-8')
                print cat               # デバッグ用プリント
                nb.train(doc, cat)
                
            else: event.Skip()

            trainDlg.Destroy()

        elif self.rb2.GetValue():
            # 分類にチェックが入ってた場合
            classifierDlg = classifierDialog(None, -1, "")
            result = classifierDlg.ShowModal()

            if result == wx.ID_OK:
                self.words = classifierDlg.text_ctrl_3.GetValue().encode('utf-8')
                print self.words               # デバッグ用プリント
            else: event.Skip()

            classifierDlg.Destroy()
        else: event.Skip()
            
    def onApply(self, event): # wxGlade: wxNaiveBayes.<event_handler>
        print u'%s => 推定カテゴリ: %s' % (self.words, nb.classifier(self.words))

    def onClose(self, event): # wxGlade: wxNaiveBayes.<event_handler>
        self.Close()

# end of class wxNaiveBayes


class trainDialog(wx.Dialog):
    def __init__(self, *args, **kwds):
        # begin wxGlade: trainDialog.__init__
        kwds["style"] = wx.DEFAULT_DIALOG_STYLE
        wx.Dialog.__init__(self, *args, **kwds)
        self.text_ctrl_1 = wx.TextCtrl(self, -1, "", style=wx.TE_MULTILINE)
        self.label_1 = wx.StaticText(self, -1, u"カテゴリ", style=wx.ALIGN_CENTRE)
        self.text_ctrl_2 = wx.TextCtrl(self, -1, "")
        self.btnOK = wx.Button(self, wx.ID_OK, "")

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: trainDialog.__set_properties
        self.SetTitle(u"文書入力(訓練用)")
        self.label_1.SetMinSize((60, 30))
        self.text_ctrl_2.SetMinSize((140, 30))
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: trainDialog.__do_layout
        sizer_4 = wx.BoxSizer(wx.VERTICAL)
        sizer_5 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_4.Add(self.text_ctrl_1, 7, wx.EXPAND, 0)
        sizer_5.Add(self.label_1, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_5.Add(self.text_ctrl_2, 0, 0, 0)
        sizer_5.Add(self.btnOK, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_4.Add(sizer_5, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_4)
        sizer_4.Fit(self)
        self.Layout()
        # end wxGlade

# end of class trainDialog


class classifierDialog(wx.Dialog):
    def __init__(self, *args, **kwds):
        # begin wxGlade: classfierDialog.__init__
        kwds["style"] = wx.DEFAULT_DIALOG_STYLE
        wx.Dialog.__init__(self, *args, **kwds)
        self.text_ctrl_3 = wx.TextCtrl(self, -1, "", style=wx.TE_MULTILINE)
        self.btnOK = wx.Button(self, wx.ID_OK, "")

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: classfierDialog.__set_properties
        self.SetTitle(u"文書入力(判別用)")
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: classfierDialog.__do_layout
        sizer_6 = wx.BoxSizer(wx.VERTICAL)
        sizer_7 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_6.Add(self.text_ctrl_3, 6, wx.EXPAND, 0)
        sizer_7.Add((250, 30), 0, 0, 0)
        sizer_7.Add(self.btnOK, 0, 0, 0)
        sizer_6.Add(sizer_7, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_6)
        sizer_6.Fit(self)
        self.Layout()
        # end wxGlade

# end of class classfierDialog


if __name__ == "__main__":
    app = wx.PySimpleApp(0)
    wx.InitAllImageHandlers()
    TopLevel = wxNaiveBayes(None, -1, "")
    app.SetTopWindow(TopLevel)
    TopLevel.Show()
    app.MainLoop()
