# Fondasi tracking RCA di ClickUp

**Catatan migrasi:** tujuan aktif kini memakai **Team Space / Problem & Incident Register**,
sesuai tautan baru pengguna. Lihat [setup Team Space](clickup_team_space.md).
Dokumen ini mencatat setup sebelumnya di Caliber Goat, bukan konfigurasi aktif.

Status per 28 September 2026: Folder dan List sudah dibuat pada Space yang
dikirim pengguna, **Caliber Goat**. Ini berada di workspace dengan ID
`1100330000013043`; tidak ada workspace tambahan yang dibuat oleh integrasi.

- [Space Caliber Goat](https://app.clickup.com/1100330000013043/v/s/1100330000053190)
- Folder RCA & Maintenance: `1100330000065685`.
- [List Investigasi RCA](https://app.clickup.com/1100330000013043/v/l/li/1100330000067880)
- Panduan kerja dan format laporan sudah tersimpan di deskripsi List.

Pembuatan dokumen terpisah ditolak dengan `not_found_or_authorized`, sehingga
materinya disimpan di deskripsi List. Tidak ada task insiden yang dibuat.
Status khusus, custom field verifikasi, dan koneksi API dashboard belum aktif.
Koneksi yang tersedia belum menyediakan pengaturan status/custom field.

## Struktur tujuan

| Tingkat | Nama | Fungsi |
| --- | --- | --- |
| Workspace | ID 1100330000013043 | Workspace dari tautan pengguna |
| Space | Caliber Goat | Space tujuan yang dipilih pengguna |
| Folder | RCA & Maintenance | Mengelompokkan pekerjaan scope 3–4 |
| List | Investigasi RCA | Satu task per kasus untuk review dan tindak lanjut |

Ringkasan deskripsi List yang sudah disimpan (deskripsi lengkap juga memuat template):

> Tracking investigasi equipment CALIBER. Draft RCA berasal dari AI Ollama
> berdasarkan data equipment dan hasil model Faiz, dengan SHAP jika tersedia.
> Task baru berstatus Need Verification. Engineer/SME memutuskan tindakan,
> mencatat hasil investigasi, dan memverifikasi RCA final. Hanya kasus yang
> selesai dan sudah diverifikasi yang dapat menjadi riwayat untuk RAG.

## Status investigasi

| Status | Makna |
| --- | --- |
| Need Verification | Draft AI menunggu pemeriksaan SME |
| Investigating | SME memeriksa bukti dan dugaan penyebab |
| Action in Progress | Tindakan yang disetujui SME sedang dikerjakan |
| Closed | Investigasi dan tindak lanjut selesai |

Atur Closed sebagai kategori penutupan ClickUp, bukan sekadar nama status aktif.
Kasus Closed yang belum memiliki RCA final dan identitas verifikator tidak boleh
masuk ke basis pengetahuan. Kasus yang tidak membuktikan dugaan AI perlu mencatat
hasil investigasi yang sebenarnya; jangan menyalin dugaan AI sebagai fakta.

Status perlu dikonfigurasi pada List tujuan. Parameter status saat membuat List
tidak dianggap sebagai pembuatan seluruh pilihan status task.

Hasil pemeriksaan List saat setup: status bawaan yang diwarisi adalah `to do`,
`planning`, `in progress`, `at risk`, `update required`, `on hold`, `complete`
(kategori done), dan `cancelled` (kategori closed). Status tersebut belum
diganti. Jangan memakai `cancelled` sebagai pengganti RCA selesai terverifikasi.
Ubah status khusus pada List ini agar tidak mengubah List lain di Space.

## Informasi task

- Judul: equipment dan ringkasan masalah.
- PIC dan prioritas memakai properti bawaan task. PIC corrective selalu dibiarkan
  kosong sampai anggota Maintenance/SME yang berwenang menetapkannya manual.
  Target waktu per tindakan dicantumkan di deskripsi sebagai usulan perencanaan.
- Isi task menyimpan snapshot, hasil model, bukti SHAP yang benar-benar tersedia,
  probable RCA, referensi, keterbatasan, dan rekomendasi pemeriksaan.
- Checklist memuat rekomendasi untuk ditinjau SME, bukan perintah operasional otomatis.
- Custom Field **RCA Final SME**: teks resolusi yang telah diverifikasi.
- Custom Field **Diverifikasi oleh**: teks nama/ID SME yang bertanggung jawab.

Pemeriksaan API saat setup menemukan nol custom field pada List baru; kedua
field tersebut masih perlu dibuat lewat aplikasi ClickUp dan ID-nya dicatat.

Board yang disarankan dikelompokkan menurut status. List menampilkan equipment,
PIC, prioritas, tenggat, dan tautan task. Pengaturan view belum diterapkan.

## Format laporan untuk isi task

### Ringkasan kondisi

Equipment: [tag equipment dari data kasus].
Waktu snapshot: [waktu sumber].
Status data: [aktual atau skenario; jangan menganggap tanggal masa depan sebagai aktual].
Ringkasan kondisi: [observasi yang benar-benar tersedia].

### Hasil model dan bukti

[Horizon dan skor prediksi Faiz yang tersedia.]

[Tiga kontribusi fitur SHAP terbesar jika berhasil dihitung, beserta arah pengaruh
dan satuan skor model. Jika belum tersedia, tuliskan alasannya. Deviasi sensor
tidak boleh diberi label sebagai nilai SHAP.]

### Dugaan penyebab dari AI

[Dugaan yang didukung bukti dan perlu diverifikasi SME. Jika belum cukup bukti,
nyatakan penyebab belum dapat dipastikan.]

### Referensi dan keterbatasan

[Rujukan insiden terverifikasi beserta lokasi/tautan, hanya jika tersedia.
Dokumen PPTX tervalidasi untuk equipment yang sama memenuhi syarat melalui exact
equipment match. Rujukan lain harus memiliki similarity >0,75; tepat 75% belum
melewati ambang. Similarity mengukur kemiripan dokumen, bukan probabilitas penyebab benar.]

[Lima PPTX RCA/CAPA merupakan evidence historis tervalidasi. Gunakan temuan RCA dan
CAPA yang relevan sebagai precedent, terutama untuk equipment yang sama, tetapi jangan
menganggap penyebab historis otomatis menjadi penyebab kondisi saat ini.]

### Rekomendasi pemeriksaan

[Langkah pemeriksaan berdasarkan bukti yang tersedia, untuk ditinjau SME.]

### Tindakan, target waktu, dan KPI corrective

[Tiga tindakan wajib ditulis terpisah: containment/verifikasi segera, corrective action
setelah penyebab dikonfirmasi, dan recurrence prevention. Satu target waktu relatif dan
satu KPI outcome diberikan untuk setiap tindakan. Target boleh
diusulkan secara reasonable dalam hitungan jam atau hari. KPI tidak mensyaratkan lampiran
atau bukti penyelesaian. Gunakan baseline/limit resmi site dan monitoring window yang
disetujui SME; jangan mengarang angka batas operasi.]

### Hasil investigasi SME

Belum diisi. SME memperbarui hasil pemeriksaan di ClickUp dan mengisi RCA Final
SME serta Diverifikasi oleh setelah menyetujui resolusi. Draft AI tetap
dibedakan dari hasil akhir ini.

## Koneksi dengan dashboard lokal

Koneksi ClickUp dalam percakapan tidak otomatis menyediakan token untuk Streamlit.
Konfigurasi runtime berikut diisi di `.env` atau `.streamlit/secrets.toml`, bukan
di kode atau percakapan. ID harus diambil dari objek yang benar-benar dibuat.

| Konfigurasi | Nilai yang diperlukan |
| --- | --- |
| CLICKUP_API_TOKEN | Token lokal yang memiliki akses ke workspace tujuan |
| CLICKUP_LIST_ID | 1100330000067880 |
| CLICKUP_REVIEW_STATUS | Need Verification |
| CLICKUP_SOLVED_STATUSES | Nama status final yang ditampilkan sebagai Solved, dipisahkan koma; contoh `done,complete,closed` |
| CLICKUP_VERIFIED_STATUS | Closed, sesuai status penutupan List ini |
| CLICKUP_VERIFIED_RCA_FIELD_ID | ID field teks RCA Final SME |
| CLICKUP_VERIFIED_BY_FIELD_ID | ID field teks Diverifikasi oleh |
| CLICKUP_ASSIGNEE_IDS | ID anggota Maintenance/SME yang dikonfirmasi, boleh kosong saat setup |

Konfigurasi lokal pada setup awal memakai ID List di atas. Setelah migrasi,
`.env` memakai List insiden Team Space; lihat dokumen setup aktif untuk detail.
Token dan ID field tidak dibuat-buat.

Target tampilan dashboard adalah tabel status, PIC, prioritas, tenggat, ringkasan
RCA, dan tautan task dari API. Tampilan tracking tersebut masih perlu dihubungkan
setelah tujuan dan konfigurasi tersedia. Task/RCA privat tidak perlu dipublikasikan
untuk iframe. Generasi mengutamakan Qwen lokal `qwen3.5:4b` dan memakai fallback
Ollama Cloud `gemma4:31b` hanya ketika layanan/model lokal tidak tersedia. Pemicu
dari dalam ClickUp memerlukan layanan penghubung tambahan dan belum menjadi bagian setup awal ini.

Pengujian integrasi pembuatan task tetap memakai dry-run/mock, tanpa task sungguhan.

## Langkah yang masih diperlukan

1. Atur status khusus pada List Investigasi RCA sesuai tabel di atas.
2. Tambahkan dua field teks RCA Final SME dan Diverifikasi oleh.
3. Catat ID field ke konfigurasi lokal dan tentukan anggota Maintenance/SME
   untuk assignment bila diperlukan.
4. Untuk menghubungkan Streamlit, pemilik akun memasukkan token ClickUp ke `.env`
   secara lokal. Jangan kirim token melalui chat. Koneksi ClickUp dalam percakapan
   tidak otomatis mengisi token aplikasi lokal.
5. Verifikasi konfigurasi melalui API baca dan uji payload dengan dry-run/mock.
   Pembuatan task insiden sungguhan bukan bagian pengujian ini.

Panduan resmi: [Create a new Workspace](https://help.clickup.com/hc/en-us/articles/6310502590487-Create-a-new-Workspace).
