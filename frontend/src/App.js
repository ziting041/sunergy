// src/App.js
import React, { useState, useEffect } from 'react';

// ===== 基礎組件 =====
import PublicHome from './pages/PublicHome';
import LoginModal from './components/LoginModal';
import RegisterModal from './components/RegisterModal';
import Navbar from './components/Navbar'; // 確保 Navbar 存在

// ===== 新版所有頁面 =====
import Dashboard from './pages/Dashboard';
import DataVariableGuide from './pages/DataVariableGuide'; // 新增：步驟 0
import StartPredict from './pages/StartPredict';        // 步驟 1
import DataCleaning from './pages/DataCleaning';        // 步驟 2
import ModelTraining from './pages/ModelTraining';      // 步驟 4
import PredictSolar from './pages/PredictSolar';        // 步驟 5 (預測結果)
import Sites from './pages/Sites';
import ModelManagement from './pages/ModelManagement';
import UserGuide from './pages/UserGuide';

// ===== Modals =====
import CreateSiteModal from './components/CreateSiteModal';
import EditSiteModal from './components/EditSiteModal'; // 帶入後端版的編輯功能

function App() {
  // ==============================
  // 1. 狀態管理 (整合後端版)
  // ==============================
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [currentPage, setCurrentPage] = useState('home');
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [isRegisterModalOpen, setIsRegisterModalOpen] = useState(false);
  const [isCreateSiteModalOpen, setIsCreateSiteModalOpen] = useState(false);
  const [editingSite, setEditingSite] = useState(null); // 案場編輯狀態

  // 登入持久化檢查 (從 App2.js 帶入)
  useEffect(() => {
    
    // 1. 優先檢查 sessionStorage (分頁關掉就沒了)
    const sessionUser = sessionStorage.getItem("user");
    const sessionPage = sessionStorage.getItem("currentPage");

    if (sessionUser) {
      try {
        const userData = JSON.parse(sessionUser);
        setCurrentUser(userData);
        setIsLoggedIn(true);
        if (sessionPage) setCurrentPage(sessionPage);
      } catch (e) {
        sessionStorage.removeItem("user");
      }
    } else {
      // 2. 重要：如果 session 沒資料，代表是新開的分頁
      // 這時我們要強制清除 localStorage，防止舊的「持久化」資料干擾
      localStorage.clear(); 
      setIsLoggedIn(false);
      setCurrentUser(null);
      setCurrentPage('home'); // 回到首頁
    }
  }, []);

  // ==============================
  // 2. 導航函式 (整合新版架構)
  // ==============================
  const navigate = (page) => {
    setCurrentPage(page);
    sessionStorage.setItem("currentPage", page);
    window.scrollTo(0, 0);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setCurrentUser(null);
    sessionStorage.removeItem("user");
    sessionStorage.removeItem("currentPage"); 
    sessionStorage.clear();
    localStorage.clear();
    navigate("home");
  };

  const handleLoginSuccess = (user) => {
    // 1. 在寫入新使用者資訊前，先清除所有舊的緩存資料
    // 這會移除舊的 currentPage, lastDataId 以及任何先前留下的狀態
    sessionStorage.clear();

    // 2. 更新 React 狀態
    setIsLoggedIn(true);
    setCurrentUser(user);

    // 3. 儲存新的使用者登入資訊
    sessionStorage.setItem("user", JSON.stringify(user));

    // 4. 強制將頁面導向 Dashboard (首頁)，避免停留在先前的訓練流程頁面
    setIsLoginModalOpen(false);
    navigate('dashboard');
  };


  // ==============================
  // 3. API 串接函式 (從 App2.js 帶入)
  // ==============================
  const submitCreateSite = async (formData) => {
    if (!currentUser) return { success: false, message: "請先登入" };
    const res = await fetch("http://127.0.0.1:8000/site/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...formData, user_id: currentUser.user_id }),
    });
    const data = await res.json();
    if (res.ok) {
      window.dispatchEvent(new Event("site-updated"));
      return { success: true };
    } else {
      return { success: false, message: data.detail || "新增案場失敗" };
    }
  };

  // ==============================
  // 4. 渲染邏輯 (新版導航 + 後端資料)
  // ==============================
  const renderContent = () => {
    if (currentPage === 'user-guide') return <UserGuide onFinish={() => navigate('home')} />;

    if (!isLoggedIn) {
      return (
        <>
          <PublicHome 
            onOpenLogin={() => setIsLoginModalOpen(true)} 
            onOpenUserGuide={() => navigate('user-guide')} 
          />
          {isLoginModalOpen && (
            <LoginModal 
              onClose={() => setIsLoginModalOpen(false)} 
              onSwitchToRegister={() => { setIsLoginModalOpen(false); setIsRegisterModalOpen(true); }} 
              onLoginSuccess={handleLoginSuccess} 
            />
          )}
          {isRegisterModalOpen && (
            <RegisterModal 
              onClose={() => setIsRegisterModalOpen(false)} 
              onSwitchToLogin={() => { setIsRegisterModalOpen(false); setIsLoginModalOpen(true); }} 
            />
          )}
        </>
      );
    }

    // 核心：新版 Navbar Props (傳遞 activePage 給導航欄高亮)
    const commonNavbarProps = {
      activePage: currentPage,
      user: currentUser,
      onNavigateToDashboard: () => navigate('dashboard'),
      onNavigateToTrain: () => navigate('data-guide'), // 訓練流程起點
      onNavigateToPredict: () => navigate('predict-solar'),
      onNavigateToSites: () => navigate('site'),
      onNavigateToModelMgmt: () => navigate('model-mgmt'),
      onLogout: handleLogout
    };

    switch (currentPage) {
      case 'dashboard':
        return <Dashboard {...commonNavbarProps} onOpenCreateSite={() => setIsCreateSiteModalOpen(true)} />;
      
      case 'site':
        return <Sites {...commonNavbarProps} onOpenCreateSite={() => setIsCreateSiteModalOpen(true)} onOpenEditSite={setEditingSite} />;
      
      case 'model-mgmt': 
        return <ModelManagement {...commonNavbarProps} />;

      case 'predict-solar':
        return <PredictSolar {...commonNavbarProps} onBack={() => navigate('dashboard')} />;

      /* --- 訓練流程五步驟 --- */
      case 'data-guide': // (1) 說明頁
        return <DataVariableGuide {...commonNavbarProps} onBack={() => navigate('dashboard')} onNext={() => navigate('start-predict')} />;

      case 'start-predict': // (2) 上傳資料
        return <StartPredict {...commonNavbarProps} onBack={() => navigate('data-guide')} onNext={() => navigate('data-cleaning')} />;
      
      case 'data-cleaning':
        return <DataCleaning {...commonNavbarProps} 
          onBack={() => navigate('start-predict')} 
          onNext={() => navigate('model-training')} />;
            case 'model-training': // (5) 訓練
        return <ModelTraining {...commonNavbarProps} onBack={() => navigate('data-cleaning')} onNext={() => navigate('predict-solar')} />;

      default:
        return <Dashboard {...commonNavbarProps} onOpenCreateSite={() => setIsCreateSiteModalOpen(true)} />;
    }
  };

  return (
    <>
      {renderContent()}
      
      {/* 全域 Modals */}
      {isLoggedIn && isCreateSiteModalOpen && (
        <CreateSiteModal onClose={() => setIsCreateSiteModalOpen(false)} onSubmit={submitCreateSite} />
      )}
      {isLoggedIn && editingSite && (
        <EditSiteModal site={editingSite} onClose={() => setEditingSite(null)} />
      )}
    </>
  );
}

export default App;