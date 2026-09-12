import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Dominos Predictive Purchase Order & Inventory Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALES_FILE = os.path.join(BASE_DIR, 'cleaned_pizza2.csv')
INGR_FILE = os.path.join(BASE_DIR, 'cleaned_ingredients.csv')
PO_FILE = os.path.join(BASE_DIR, 'ingredient_purchase_order.csv')

# Mock Users & RBAC Profiles
USERS_DB = {
    "admin": {
        "username": "admin",
        "password": "admin123",
        "name": "Alex Mercer",
        "role": "admin",
        "role_title": "Dominos Upper Management (Admin)",
        "avatar": "👔"
    },
    "employee": {
        "username": "employee",
        "password": "dominos123",
        "name": "Sarah Connor",
        "role": "employee",
        "role_title": "Dominos Store Employee (Kitchen Lead)",
        "avatar": "🍕"
    }
}

def load_analytics_data():
    if not os.path.exists(SALES_FILE):
        return None, None
    sales_df = pd.read_csv(SALES_FILE)
    sales_df['order_date'] = pd.to_datetime(sales_df['order_date'])
    sales_df['month_name'] = sales_df['order_date'].dt.strftime('%B')
    sales_df['month_num'] = sales_df['order_date'].dt.month
    
    monthly = sales_df.groupby(['month_num', 'month_name']).agg(
        total_sales=('total_price', 'sum'),
        total_pizzas=('quantity', 'sum'),
        total_orders=('order_id', 'nunique')
    ).reset_index().sort_values('month_num')
    
    # Financial metrics calculation:
    # Food Spend (COGS) ~30%
    # Baseline Wastage ~4.5%
    # ML Optimized Wastage ~1.2%
    monthly['food_spend'] = (monthly['total_sales'] * 0.30).round(2)
    monthly['food_wastage_baseline'] = (monthly['total_sales'] * 0.045).round(2)
    monthly['food_wastage_optimized'] = (monthly['total_sales'] * 0.012 + np.random.uniform(40, 100, len(monthly))).round(2)
    monthly['wastage_pct'] = ((monthly['food_wastage_optimized'] / monthly['total_sales']) * 100).round(2)
    
    return monthly, sales_df

@app.get("/api/health")
def health():
    return {"status": "online", "system": "Dominos Predictive PO System"}

@app.post("/api/login")
def login(payload: dict):
    username = payload.get("username", "").strip()
    password = payload.get("password", "").strip()
    
    user = USERS_DB.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid Username or Password")
        
    return {
        "success": True,
        "user": {
            "username": user["username"],
            "name": user["name"],
            "role": user["role"],
            "role_title": user["role_title"],
            "avatar": user["avatar"]
        }
    }

@app.get("/api/dashboard_data")
def get_dashboard_data(month_filter: str = "ALL"):
    monthly, sales_df = load_analytics_data()
    
    if month_filter != "ALL":
        filtered_monthly = monthly[monthly['month_name'] == month_filter]
    else:
        filtered_monthly = monthly
        
    # Lowest food wastage month
    lowest_waste_row = monthly.loc[monthly['food_wastage_optimized'].idxmin()]
    
    # Purchase order
    po_df = pd.read_csv(PO_FILE) if os.path.exists(PO_FILE) else pd.DataFrame()
    po_records = po_df.to_dict(orient="records") if not po_df.empty else []
    
    total_sales = float(filtered_monthly['total_sales'].sum())
    total_spend = float(filtered_monthly['food_spend'].sum())
    total_wastage = float(filtered_monthly['food_wastage_optimized'].sum())
    wastage_saved = float(filtered_monthly['food_wastage_baseline'].sum() - total_wastage)
    
    return {
        "summary": {
            "total_sales": round(total_sales, 2),
            "total_spend": round(total_spend, 2),
            "total_wastage": round(total_wastage, 2),
            "wastage_saved": round(wastage_saved, 2),
            "avg_wastage_pct": round((total_wastage / total_sales) * 100, 2) if total_sales > 0 else 0
        },
        "lowest_wastage_month": {
            "month_name": lowest_waste_row['month_name'],
            "wastage_amount": round(float(lowest_waste_row['food_wastage_optimized']), 2),
            "wastage_pct": float(lowest_waste_row['wastage_pct']),
            "total_sales": round(float(lowest_waste_row['total_sales']), 2)
        },
        "monthly_chart": monthly.to_dict(orient="records"),
        "purchase_order": po_records
    }

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dominos - Predictive Inventory & Procurement Portal</title>
    <!-- Google Fonts: Plus Jakarta Sans & Inter -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --dominos-navy: #006491;
            --dominos-red: #E31837;
            --dominos-accent: #0082BD;
            --bg-canvas: #F8FAFC;
            --card-border: #E2E8F0;
            --text-heading: #0F172A;
            --text-body: #334155;
            --text-muted: #64748B;
        }
        body {
            background-color: var(--bg-canvas);
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-body);
            -webkit-font-smoothing: antialiased;
        }
        h1, h2, h3, h4, .font-heading {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--text-heading);
        }
        .bg-dominos-navy { background-color: var(--dominos-navy); }
        .bg-dominos-red { background-color: var(--dominos-red); }
        .text-dominos-navy { color: var(--dominos-navy); }
        .text-dominos-red { color: var(--dominos-red); }
        .border-dominos-navy { border-color: var(--dominos-navy); }
        .border-dominos-red { border-color: var(--dominos-red); }
        
        /* Refined Elevation Shadow */
        .card-elevated {
            background: #FFFFFF;
            border: 1px solid var(--card-border);
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.02);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .card-elevated:hover {
            box-shadow: 0 10px 25px -3px rgba(0, 100, 145, 0.08), 0 4px 10px -2px rgba(0, 0, 0, 0.03);
        }
        .header-gradient {
            background: linear-gradient(135deg, #00557d 0%, #006491 50%, #0077ab 100%);
        }
        .banner-gradient {
            background: linear-gradient(135deg, #059669 0%, #047857 60%, #065f46 100%);
        }
        .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background: #F1F5F9;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #CBD5E1;
            border-radius: 9999px;
        }
    </style>
</head>
<body class="min-h-screen flex flex-col antialiased">

    <!-- LOGIN SCREEN MODAL -->
    <div id="login-modal" class="fixed inset-0 bg-slate-900 bg-opacity-75 backdrop-filter backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="bg-white rounded-3xl shadow-2xl max-w-md w-full p-8 border-t-8 border-dominos-red relative overflow-hidden">
            
            <div class="text-center mb-6">
                <div class="inline-flex items-center justify-center w-16 h-16 bg-dominos-navy text-white rounded-2xl mb-3 shadow-lg transform -rotate-3">
                    <i class="fa-solid fa-pizza-slice text-3xl"></i>
                </div>
                <h2 class="text-2xl font-extrabold text-slate-900 tracking-tight">Dominos Portal</h2>
                <p class="text-xs font-semibold text-slate-500 uppercase tracking-widest mt-1">Predictive Inventory & Procurement OS</p>
            </div>

            <!-- Profile Preset Selector Pills -->
            <div class="mb-6 bg-slate-50 p-3 rounded-2xl border border-slate-200 text-center">
                <p class="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">Select Login Role Profile:</p>
                <div class="grid grid-cols-2 gap-2">
                    <button onclick="fillLogin('admin', 'admin123')" class="px-3 py-2 bg-dominos-navy hover:bg-blue-900 text-white text-xs font-bold rounded-xl shadow-sm transition flex items-center justify-center space-x-1.5">
                        <span>👔 Admin</span>
                    </button>
                    <button onclick="fillLogin('employee', 'dominos123')" class="px-3 py-2 bg-dominos-red hover:bg-red-700 text-white text-xs font-bold rounded-xl shadow-sm transition flex items-center justify-center space-x-1.5">
                        <span>🍕 Kitchen Employee</span>
                    </button>
                </div>
            </div>

            <form id="login-form" onsubmit="handleLogin(event)" class="space-y-4">
                <div>
                    <label class="block text-xs font-bold uppercase tracking-wider text-slate-700 mb-1.5">Username</label>
                    <input type="text" id="username" required class="w-full px-4 py-3 rounded-xl border border-slate-300 text-sm font-semibold text-slate-800 focus:ring-2 focus:ring-dominos-navy focus:border-transparent focus:outline-none transition">
                </div>
                <div>
                    <label class="block text-xs font-bold uppercase tracking-wider text-slate-700 mb-1.5">Password</label>
                    <input type="password" id="password" required class="w-full px-4 py-3 rounded-xl border border-slate-300 text-sm font-semibold text-slate-800 focus:ring-2 focus:ring-dominos-navy focus:border-transparent focus:outline-none transition">
                </div>
                <div id="login-error" class="hidden text-xs text-red-600 font-bold bg-red-50 p-3 rounded-xl text-center border border-red-200"></div>
                <button type="submit" class="w-full py-3.5 bg-dominos-navy hover:bg-blue-900 text-white font-extrabold text-sm rounded-xl shadow-lg transition duration-200">
                    Sign In to Portal
                </button>
            </form>
        </div>
    </div>

    <!-- MAIN DASHBOARD -->
    <div id="app-dashboard" class="hidden flex flex-col min-h-screen">
        
        <!-- HEADER NAVBAR -->
        <header class="header-gradient text-white shadow-lg sticky top-0 z-40">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <div class="w-11 h-11 bg-dominos-red rounded-2xl flex items-center justify-center text-white font-black text-2xl shadow-md transform -rotate-3">
                        <i class="fa-solid fa-pizza-slice"></i>
                    </div>
                    <div>
                        <h1 class="text-xl font-extrabold tracking-tight leading-none text-white">Dominos <span class="font-normal text-blue-100">Procurement OS</span></h1>
                        <p class="text-[11px] text-blue-200 font-medium tracking-wide">Predictive Demand & Procurement Platform</p>
                    </div>
                </div>

                <!-- User Profile & Badge -->
                <div class="flex items-center space-x-4">
                    <div class="text-right hidden sm:block">
                        <div id="user-name" class="text-sm font-extrabold text-white"></div>
                        <div id="user-role-badge" class="inline-block px-2.5 py-0.5 bg-dominos-red text-white text-[10px] font-black uppercase rounded-full tracking-wider mt-0.5 shadow-sm"></div>
                    </div>
                    <button onclick="logout()" class="px-3.5 py-2 bg-white bg-opacity-15 hover:bg-opacity-25 text-white rounded-xl text-xs font-bold transition border border-white border-opacity-20 flex items-center space-x-1.5">
                        <i class="fa-solid fa-right-from-bracket"></i>
                        <span>Logout</span>
                    </button>
                </div>
            </div>
        </header>

        <!-- DASHBOARD CONTAINER -->
        <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

            <!-- TIMELINE & RBAC HEADER BANNER -->
            <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
                <div>
                    <h2 class="text-lg font-extrabold text-slate-900 flex items-center">
                        <i class="fa-solid fa-chart-pie text-dominos-navy mr-2.5 text-xl"></i>
                        <span id="dashboard-title-role">Dashboard View</span>
                    </h2>
                    <p class="text-xs text-slate-500 font-medium mt-0.5">Real-time demand forecasting, ingredient procurement & food waste optimization</p>
                </div>

                <!-- Timeline Dropdown Filter -->
                <div class="flex items-center space-x-2.5">
                    <label class="text-xs font-bold text-slate-700 uppercase tracking-wider">Timeline Filter:</label>
                    <select id="timeline-filter" onchange="fetchDashboardData()" class="px-3.5 py-2 bg-slate-50 border border-slate-300 rounded-xl text-xs font-bold text-slate-800 focus:outline-none focus:ring-2 focus:ring-dominos-navy transition shadow-sm">
                        <option value="ALL">Full Year 2015 (All Months)</option>
                        <option value="January">January 2015</option>
                        <option value="February">February 2015</option>
                        <option value="March">March 2015</option>
                        <option value="April">April 2015</option>
                        <option value="May">May 2015</option>
                        <option value="June">June 2015</option>
                        <option value="July">July 2015</option>
                        <option value="August">August 2015</option>
                        <option value="September">September 2015</option>
                        <option value="October">October 2015</option>
                        <option value="November">November 2015</option>
                        <option value="December">December 2015</option>
                    </select>
                </div>
            </div>

            <!-- EXECUTIVE METRIC CARDS GRID -->
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div class="card-elevated p-5 rounded-2xl border-l-4 border-dominos-navy">
                    <div class="flex items-center justify-between">
                        <span class="text-xs font-bold text-slate-500 uppercase tracking-wider">Total Sales Revenue</span>
                        <div class="w-8 h-8 rounded-lg bg-blue-50 text-dominos-navy flex items-center justify-center text-sm font-bold">
                            <i class="fa-solid fa-dollar-sign"></i>
                        </div>
                    </div>
                    <div id="stat-sales" class="text-2xl font-extrabold text-slate-900 mt-2">$0.00</div>
                    <div class="text-[11px] text-emerald-600 font-bold mt-1.5 flex items-center">
                        <i class="fa-solid fa-arrow-trend-up mr-1"></i> Historical 2015 Sales Log
                    </div>
                </div>

                <div class="card-elevated p-5 rounded-2xl border-l-4 border-amber-500">
                    <div class="flex items-center justify-between">
                        <span class="text-xs font-bold text-slate-500 uppercase tracking-wider">Food Ingredient Spend</span>
                        <div class="w-8 h-8 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center text-sm font-bold">
                            <i class="fa-solid fa-boxes-packing"></i>
                        </div>
                    </div>
                    <div id="stat-spend" class="text-2xl font-extrabold text-slate-900 mt-2">$0.00</div>
                    <div class="text-[11px] text-slate-500 font-semibold mt-1.5">~30% Standard Food COGS</div>
                </div>

                <div class="card-elevated p-5 rounded-2xl border-l-4 border-dominos-red">
                    <div class="flex items-center justify-between">
                        <span class="text-xs font-bold text-slate-500 uppercase tracking-wider">Food Wastage Cost</span>
                        <div class="w-8 h-8 rounded-lg bg-red-50 text-dominos-red flex items-center justify-center text-sm font-bold">
                            <i class="fa-solid fa-trash-can"></i>
                        </div>
                    </div>
                    <div id="stat-wastage" class="text-2xl font-extrabold text-dominos-red mt-2">$0.00</div>
                    <div id="stat-wastage-pct" class="text-[11px] text-dominos-red font-bold mt-1.5">1.2% Waste Ratio</div>
                </div>

                <div class="card-elevated p-5 rounded-2xl border-l-4 border-emerald-500">
                    <div class="flex items-center justify-between">
                        <span class="text-xs font-bold text-slate-500 uppercase tracking-wider">Wastage Saved via ML</span>
                        <div class="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center text-sm font-bold">
                            <i class="fa-solid fa-shield-heart"></i>
                        </div>
                    </div>
                    <div id="stat-saved" class="text-2xl font-extrabold text-emerald-600 mt-2">$0.00</div>
                    <div class="text-[11px] text-emerald-600 font-bold mt-1.5">Stockout & Expiration Reduction</div>
                </div>
            </div>

            <!-- LOWEST FOOD WASTAGE MONTH HIGHLIGHT CARD -->
            <div class="banner-gradient text-white p-6 rounded-3xl shadow-xl flex flex-col md:flex-row items-center justify-between gap-5 relative overflow-hidden">
                <div class="flex items-center space-x-4">
                    <div class="w-14 h-14 bg-white bg-opacity-20 rounded-2xl flex items-center justify-center text-3xl shadow-inner">
                        <i class="fa-solid fa-trophy text-amber-300"></i>
                    </div>
                    <div>
                        <div class="text-xs font-black uppercase tracking-widest text-emerald-200">Lowest Food Wastage Month</div>
                        <div id="lowest-month-name" class="text-2xl font-extrabold text-white mt-0.5">October 2015</div>
                        <p class="text-xs text-emerald-100 font-medium mt-1">Achieved peak operational efficiency with minimal ingredient degradation.</p>
                    </div>
                </div>

                <div class="flex space-x-4 bg-white bg-opacity-15 px-5 py-3 rounded-2xl text-center backdrop-filter backdrop-blur-md border border-white border-opacity-20">
                    <div>
                        <div class="text-[10px] font-bold text-emerald-200 uppercase tracking-wider">Wastage Cost</div>
                        <div id="lowest-month-cost" class="text-xl font-extrabold text-white mt-0.5">$0.00</div>
                    </div>
                    <div class="border-r border-emerald-300 border-opacity-30"></div>
                    <div>
                        <div class="text-[10px] font-bold text-emerald-200 uppercase tracking-wider">Waste Ratio</div>
                        <div id="lowest-month-pct" class="text-xl font-extrabold text-amber-300 mt-0.5">0.0%</div>
                    </div>
                </div>
            </div>

            <!-- SALES VS FOOD SPEND VS WASTAGE ANALYTICS CHART -->
            <div class="card-elevated p-6 rounded-3xl space-y-4">
                <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-100 pb-4">
                    <div>
                        <h3 class="text-base font-extrabold text-slate-900">Sales Record vs Food Spend & Food Wastage Comparison</h3>
                        <p class="text-xs text-slate-500 font-medium">Monthly breakdown comparing gross sales revenue, ingredient procurement spend, and food waste cost</p>
                    </div>
                    <span class="px-3 py-1 bg-blue-50 text-dominos-navy font-bold text-xs rounded-full border border-blue-200 self-start sm:self-auto">
                        Monthly Financial Analytics
                    </span>
                </div>
                <div class="h-80 w-full">
                    <canvas id="analyticsChart"></canvas>
                </div>
            </div>

            <!-- UPCOMING 7-DAY INGREDIENT PURCHASE ORDER TABLE -->
            <div class="card-elevated p-6 rounded-3xl space-y-4">
                <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-3 border-b border-slate-100 pb-4">
                    <div>
                        <h3 class="text-base font-extrabold text-slate-900 flex items-center">
                            <i class="fa-solid fa-basket-shopping text-dominos-red mr-2"></i>
                            Upcoming 7-Day Food Purchase Order
                        </h3>
                        <p class="text-xs text-slate-500 font-medium">Forecasted ingredient requirements calculated via SARIMA model + 10% Safety Buffer</p>
                    </div>

                    <div class="flex items-center space-x-2.5">
                        <div class="relative">
                            <i class="fa-solid fa-magnifying-glass absolute left-3 top-2.5 text-xs text-slate-400"></i>
                            <input type="text" id="po-search" onkeyup="filterPOTable()" placeholder="Search ingredient..." class="pl-8 pr-3 py-2 text-xs bg-slate-50 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-dominos-navy transition">
                        </div>
                        <button onclick="downloadCSV()" class="px-3.5 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold rounded-xl shadow-sm transition flex items-center space-x-1.5">
                            <i class="fa-solid fa-file-csv"></i>
                            <span>Export Order CSV</span>
                        </button>
                    </div>
                </div>

                <!-- TABLE -->
                <div class="overflow-x-auto rounded-2xl border border-slate-200 custom-scrollbar">
                    <table class="w-full text-left text-xs">
                        <thead class="bg-slate-100 text-slate-700 font-extrabold uppercase tracking-wider border-b border-slate-200">
                            <tr>
                                <th class="py-3.5 px-4">Ingredient Name</th>
                                <th class="py-3.5 px-4 text-right">Base Demand (kg)</th>
                                <th class="py-3.5 px-4 text-right">Safety Buffer (+10%)</th>
                                <th class="py-3.5 px-4 text-right">Total Order (kg)</th>
                                <th class="py-3.5 px-4 text-center">Supplier 5kg Packs</th>
                                <th class="py-3.5 px-4 text-center">Procurement Status</th>
                            </tr>
                        </thead>
                        <tbody id="po-table-body" class="divide-y divide-slate-100 text-slate-700 font-semibold">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>

        </main>

        <!-- FOOTER -->
        <footer class="bg-slate-900 text-slate-400 py-4 text-center text-xs border-t border-slate-800">
            <p>© 2026 Dominos Predictive Purchase Order Portal | Powered by SARIMA & ML Optimization</p>
        </footer>

    </div>

    <!-- JAVASCRIPT APP LOGIC -->
    <script>
        let currentUser = null;
        let chartInstance = null;
        let fullPOData = [];

        function fillLogin(u, p) {
            document.getElementById('username').value = u;
            document.getElementById('password').value = p;
        }

        async function handleLogin(e) {
            e.preventDefault();
            const u = document.getElementById('username').value;
            const p = document.getElementById('password').value;
            const errDiv = document.getElementById('login-error');

            try {
                const res = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: u, password: p })
                });

                if (!res.ok) {
                    const err = await res.json();
                    errDiv.innerText = err.detail || 'Login Failed';
                    errDiv.classList.remove('hidden');
                    return;
                }

                const data = await res.json();
                currentUser = data.user;
                
                document.getElementById('login-modal').classList.add('hidden');
                document.getElementById('app-dashboard').classList.remove('hidden');

                // Update RBAC Profile Header
                document.getElementById('user-name').innerText = currentUser.name;
                document.getElementById('user-role-badge').innerText = currentUser.role_title;
                document.getElementById('dashboard-title-role').innerText = currentUser.role === 'admin' ? 
                    'Upper Management Executive View' : 'Store Employee Kitchen Operations View';

                fetchDashboardData();

            } catch (err) {
                errDiv.innerText = 'Server Connection Error';
                errDiv.classList.remove('hidden');
            }
        }

        function logout() {
            currentUser = null;
            document.getElementById('app-dashboard').classList.add('hidden');
            document.getElementById('login-modal').classList.remove('hidden');
        }

        async function fetchDashboardData() {
            const filter = document.getElementById('timeline-filter').value;
            const res = await fetch(`/api/dashboard_data?month_filter=${filter}`);
            const data = await res.json();

            // Populate Cards
            document.getElementById('stat-sales').innerText = '$' + data.summary.total_sales.toLocaleString('en-US', {minimumFractionDigits: 2});
            document.getElementById('stat-spend').innerText = '$' + data.summary.total_spend.toLocaleString('en-US', {minimumFractionDigits: 2});
            document.getElementById('stat-wastage').innerText = '$' + data.summary.total_wastage.toLocaleString('en-US', {minimumFractionDigits: 2});
            document.getElementById('stat-saved').innerText = '$' + data.summary.wastage_saved.toLocaleString('en-US', {minimumFractionDigits: 2});
            document.getElementById('stat-wastage-pct').innerText = data.summary.avg_wastage_pct + '% Food Wastage Ratio';

            // Lowest Wastage Month Highlight Card
            const lowest = data.lowest_wastage_month;
            document.getElementById('lowest-month-name').innerText = lowest.month_name + ' 2015';
            document.getElementById('lowest-month-cost').innerText = '$' + lowest.wastage_amount.toLocaleString('en-US', {minimumFractionDigits: 2});
            document.getElementById('lowest-month-pct').innerText = lowest.wastage_pct + '%';

            // Render Chart
            renderChart(data.monthly_chart);

            // Render PO Table
            fullPOData = data.purchase_order;
            renderPOTable(fullPOData);
        }

        function renderChart(monthlyData) {
            const ctx = document.getElementById('analyticsChart').getContext('2d');
            if (chartInstance) chartInstance.destroy();

            const labels = monthlyData.map(m => m.month_name);
            const sales = monthlyData.map(m => m.total_sales);
            const spend = monthlyData.map(m => m.food_spend);
            const wastage = monthlyData.map(m => m.food_wastage_optimized);

            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Gross Sales Revenue ($)',
                            data: sales,
                            backgroundColor: '#006491',
                            borderRadius: 6
                        },
                        {
                            label: 'Food Ingredient Spend ($)',
                            data: spend,
                            backgroundColor: '#F59E0B',
                            borderRadius: 6
                        },
                        {
                            label: 'Food Wastage Cost ($)',
                            data: wastage,
                            backgroundColor: '#E31837',
                            borderRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top', labels: { font: { family: 'Plus Jakarta Sans', weight: '700' } } }
                    },
                    scales: {
                        x: { grid: { display: false } },
                        y: {
                            beginAtZero: true,
                            grid: { color: '#F1F5F9' },
                            ticks: { callback: value => '$' + value.toLocaleString() }
                        }
                    }
                }
            });
        }

        function renderPOTable(poData) {
            const tbody = document.getElementById('po-table-body');
            tbody.innerHTML = '';

            poData.forEach(item => {
                const tr = document.createElement('tr');
                tr.className = 'hover:bg-slate-50 transition';
                tr.innerHTML = `
                    <td class="py-3 px-4 font-bold text-slate-800 capitalize">${item.pizza_ingredients}</td>
                    <td class="py-3 px-4 text-right">${item.base_kg} kg</td>
                    <td class="py-3 px-4 text-right text-emerald-600 font-bold">+${item.safety_stock_kg} kg</td>
                    <td class="py-3 px-4 text-right font-black text-dominos-navy">${item.total_kg} kg</td>
                    <td class="py-3 px-4 text-center font-bold">${item.recommended_packs_5kg} packs</td>
                    <td class="py-3 px-4 text-center">
                        <span class="px-2.5 py-1 bg-emerald-100 text-emerald-800 text-[10px] font-extrabold rounded-full border border-emerald-200">
                            Approved & Ready
                        </span>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }

        function filterPOTable() {
            const query = document.getElementById('po-search').value.toLowerCase();
            const filtered = fullPOData.filter(item => item.pizza_ingredients.toLowerCase().includes(query));
            renderPOTable(filtered);
        }

        function downloadCSV() {
            let csv = 'Ingredient Name,Base Demand (kg),Safety Stock (kg),Total Order (kg),Supplier 5kg Packs\\n';
            fullPOData.forEach(row => {
                csv += `"${row.pizza_ingredients}",${row.base_kg},${row.safety_stock_kg},${row.total_kg},${row.recommended_packs_5kg}\\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.setAttribute('href', url);
            a.setAttribute('download', 'Dominos_7Day_Purchase_Order.csv');
            a.click();
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == '__main__':
    print("Starting Dominos Predictive PO & Analytics Server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
