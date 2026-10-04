# Tracking RCA di Team Space

Tujuan aktif dipilih pengguna pada 28 September 2026. Tautan yang dikirim adalah
Space di workspace yang sama, bukan workspace lain. Konfigurasi lokal telah
dialihkan ke struktur yang sudah ada.

| Bagian | Lokasi dan peran |
| --- | --- |
| Space | [Team Space](https://app.clickup.com/1100330000013043/v/s/1100330000037854) |
| Folder | RCA & Action Management (`1100330000057836`) |
| Workflow RCA & CAPA | [List](https://app.clickup.com/1100330000013043/v/b/li/1100330000081187) (`1100330000081187`) |

Progress Tracking hanya menulis ke satu List tersebut. List lama tetap ada tetapi
tidak digunakan untuk pembuatan task RCA/CAPA dari Executive Summary.

## Temuan konfigurasi

API mengonfirmasi urutan status List: `to review`, `to do`, `in progres`, dan
`done`. Semua task RCA/CAPA baru masuk ke `to review`.

Ada satu insiden KO-3201 berlabel DEMO dan enam tindakan CAPA terkait. Struktur
penulisannya menjadi acuan pengaturan laporan. Angka, penyebab, batas operasi,
kerugian dan tindakan dalam contoh tersebut tidak disalin ke analisis baru.
Selesainya sebuah tindakan CAPA tidak dianggap sebagai RCA insiden terverifikasi.

## Alur hasil AI

1. Data kasus tetap snapshot sensor dan hasil model Faiz yang tersedia di repo.
2. SHAP dihitung hanya ketika fitur lengkap untuk equipment/waktu itu tersedia.
3. Hanya riwayat insiden terverifikasi dengan cosine similarity >0,75 menjadi
   konteks RCA; angka persen dihitung dari skor 0–1 dikali 100.
4. Qwen lokal `qwen3.5:4b` menghasilkan draft dengan format terstruktur. Jika layanan/model
   lokal tidak tersedia, backend memakai fallback Ollama Cloud `gemma4:31b` dengan
   `OLLAMA_API_KEY` dari environment atau `.env` yang diabaikan Git.
5. Dashboard, unduhan laporan, dan deskripsi task menggunakan satu teks laporan.
6. Tombol **Create Progress Tracking** pada Executive Summary membuat satu task
   corrective gabungan yang memuat draft RCA dan seluruh usulan CAPA. Task dimulai
   pada `to review`.
7. SME meninjau RCA dan tiga tahap CAPA: containment/verifikasi segera, corrective
   action setelah penyebab dikonfirmasi, dan recurrence prevention. Task corrective
   sengaja dibuat tanpa PIC agar penanggung jawab ditetapkan manual, sedangkan target
   waktu dan KPI yang reasonable dicantumkan sebagai usulan. Setelah action disetujui, SME
   menetapkan PIC; PIC memindahkan action yang disetujui ke `to do` dan mengubahnya menjadi
   `in progres` saat pekerjaan dimulai dan `done` setelah selesai.
8. Pembuatan task dikunci ke workspace `1100330000013043`, space
   `1100330000037854`, folder `1100330000057836`, dan List `1100330000081187`.
   Konfigurasi atau respons API yang menunjuk ke lokasi lain ditolak.
9. Setelah insiden `complete` dan kedua field verifikasi terisi oleh SME,
   sinkronisasi boleh memasukkan resolusi final ke basis pengetahuan.

## Konsistensi laporan

- Ringkasan kasus, bukti model/SHAP, dugaan RCA, referensi, rekomendasi, keterbatasan,
  verifikasi SME, dan tindak lanjut CAPA memakai formatter yang sama.
- Format jawaban AI divalidasi: paragraf harus berupa teks, rekomendasi berupa
  daftar kalimat, dan field yang tidak dikenal ditolak.
- Arsip hasil disimpan di `data/09_erika/analyses/` (diabaikan Git). Input, bukti,
  nama model, prompt, versi schema dan opsi generasi menentukan ID analisis.
- Input yang sama memakai arsip yang sama, bahkan saat Ollama sedang offline.
  Input/bukti berubah menghasilkan ID baru. Arsip rusak ditolak secara jelas.
- `temperature=0` dan `seed=42` mengurangi variasi generasi. Konsistensi antar
  tampilan berasal dari memakai arsip yang sama, bukan jaminan bahwa LLM selalu benar.
- Tanpa precedent/SOP, dugaan tetap belum dapat dipastikan dan rekomendasi dibatasi
  pada validasi sensor, peninjauan data yang tersedia dan konsultasi SME.
- Skor dan ambang Faiz ditulis langsung dari data. Dampak produksi, downtime,
  PIC dan batas operasi yang tidak tersedia tidak dibuat-buat. Khusus corrective,
  target waktu relatif dan KPI boleh diusulkan secara wajar sebagai asumsi perencanaan.
- Ringkasan kondisi dan batas bukti juga ditulis dari data/aturan yang diketahui.
  Respons mentah model disimpan terpisah untuk audit. Dalam uji lokal, Qwen pernah
  menulis skor skala 0–100 sebagai persentase risiko; kalimat tersebut tidak dipakai
  dalam laporan final. Tanpa precedent, dugaan dan guidance mengikuti fallback
  terbatas, bukan asumsi Qwen tentang kegagalan fisik.
- Snapshot 4 Oktober 2026 diberi label skenario/masa depan terhadap acuan
  28 September 2026. Draft tidak menjadi pengetahuan historis sebelum verifikasi.
- Nama model adalah bagian fingerprint; penggantian isi model dengan tag yang
  sama tidak membatalkan arsip lama secara otomatis. Arsip menjaga hasil review
  sebelumnya tetap stabil, bukan klaim rerun lintas versi model identik.

## Konfigurasi lokal

Nilai nonrahasia berikut sudah disimpan di `.env` lokal:

| Variabel | Nilai |
| --- | --- |
| CLICKUP_WORKSPACE_ID | 1100330000013043 |
| CLICKUP_SPACE_ID | 1100330000037854 |
| Folder yang diwajibkan kode | 1100330000057836 |
| CLICKUP_LIST_ID | 1100330000081187 |
| CLICKUP_REVIEW_STATUS | to review |
| CLICKUP_SOLVED_STATUSES | done,complete,closed |
| CLICKUP_VERIFIED_STATUS | complete |

Token API dibaca dari `.env` lokal yang diabaikan Git. Dialog berhenti sebelum
generasi AI atau pembuatan task jika token, hierarchy, List, atau status awal tidak
sesuai. Pengujian otomatis tidak membuat task ClickUp sungguhan.

Yang perlu disiapkan melalui aplikasi ClickUp:

1. Tambahkan dua field teks **RCA Final SME** dan **Diverifikasi oleh** di List itu.
2. Simpan ID field pada `CLICKUP_VERIFIED_RCA_FIELD_ID` dan
   `CLICKUP_VERIFIED_BY_FIELD_ID`; masukkan token sendiri di `.env`, bukan chat.
3. Assignment baru diisi setelah ID anggota Maintenance/SME dikonfirmasi.

Sebelum POST, integrasi membaca tujuan List/Space dan memeriksa status yang diminta.
Jika status belum ada, task tidak dibuat dan arsip lokal dipertahankan. Log task
menyertakan List ID; pindah tujuan tidak memakai task dari List lama. Draft berbeda
untuk task/snapshot yang sudah ada memerlukan pemeriksaan, bukan penggantian diam-diam.

Fondasi lama di Caliber Goat tetap tersedia sebagai hasil setup sebelumnya.
Tidak ada penghapusan, pemindahan task lama, atau push GitHub dalam migrasi ini.

Referensi implementasi: [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
dan [ClickUp Create Task](https://developer.clickup.com/reference/createtask).
